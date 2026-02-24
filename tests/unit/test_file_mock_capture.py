"""
FileMockCapture 단위 테스트

검증 항목:
- AudioPacket 생성 및 큐 전달 확인
- 패킷 간 시간 간격이 chunk_size_ms ±허용범위 내인지 확인
- VideoFrame 생성 확인
- get_audio_queue / get_video_queue 인터페이스 정상 동작
- stop() 이후 태스크 정상 종료
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.capture import AudioPacket, VideoFrame
from src.capture.file_mock_capture import FileMockCapture
from src.config.schema import AppConfig


# =============================================================================
# 테스트 픽스처
# =============================================================================

def _make_wav_file(tmp_path: Path, duration_sec: float = 3.0, sample_rate: int = 48000) -> Path:
    """
    테스트용 WAV 파일을 생성합니다.

    48kHz/16bit/stereo, 1kHz 사인파를 duration_sec 동안 생성합니다.
    """
    wav_path = tmp_path / "test_audio.wav"
    total_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    # 1kHz 사인파, 진폭 0.5 (int16 범위 내)
    sine_wave = (np.sin(2 * np.pi * 1000 * t) * 16384).astype(np.int16)
    # 스테레오 변환 (L/R 동일)
    stereo = np.column_stack([sine_wave, sine_wave])
    sf.write(str(wav_path), stereo, sample_rate, subtype="PCM_16")
    return wav_path


def _make_config(
    tmp_path: Path,
    audio_path: str,
    chunk_size_ms: int = 100,
    loop: bool = False,
    playback_speed: float = 1.0,
) -> AppConfig:
    """테스트용 AppConfig를 생성합니다."""
    return AppConfig(**{
        "system": {"mode": "file"},
        "capture": {
            "audio_queue_size": 200,
            "video_queue_size": 60,
            "test_file": {
                "audio_path": audio_path,
                "loop": loop,
                "playback_speed": playback_speed,
            },
        },
        "audio": {
            "chunk_size_ms": chunk_size_ms,
        },
    })


# =============================================================================
# 인터페이스 테스트
# =============================================================================

def test_get_audio_queue_returns_asyncio_queue(tmp_path):
    """get_audio_queue()가 asyncio.Queue를 반환하는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path)
    config = _make_config(tmp_path, str(wav_path))
    capture = FileMockCapture(config)

    queue = capture.get_audio_queue()

    assert isinstance(queue, asyncio.Queue)


def test_get_video_queue_returns_asyncio_queue(tmp_path):
    """get_video_queue()가 asyncio.Queue를 반환하는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path)
    config = _make_config(tmp_path, str(wav_path))
    capture = FileMockCapture(config)

    queue = capture.get_video_queue()

    assert isinstance(queue, asyncio.Queue)


def test_same_queue_instance_returned(tmp_path):
    """매번 동일한 큐 인스턴스를 반환하는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path)
    config = _make_config(tmp_path, str(wav_path))
    capture = FileMockCapture(config)

    assert capture.get_audio_queue() is capture.get_audio_queue()
    assert capture.get_video_queue() is capture.get_video_queue()


# =============================================================================
# 오디오 패킷 생성 테스트
# =============================================================================

@pytest.mark.asyncio
async def test_audio_packets_generated(tmp_path):
    """오디오 패킷이 큐에 정상 생성되는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path, duration_sec=2.0)
    # 10배속으로 실행하여 테스트 시간 단축 (200ms 내 완료)
    config = _make_config(tmp_path, str(wav_path), chunk_size_ms=100, playback_speed=10.0)
    capture = FileMockCapture(config)

    await capture.start()
    # 10개 패킷 수집 (10개 × 10ms(100ms/10배속) = 100ms)
    packets = []
    try:
        for _ in range(10):
            packet = await asyncio.wait_for(
                capture.get_audio_queue().get(), timeout=2.0
            )
            packets.append(packet)
    finally:
        await capture.stop()

    assert len(packets) == 10
    assert all(isinstance(p, AudioPacket) for p in packets)


@pytest.mark.asyncio
async def test_audio_packet_fields_valid(tmp_path):
    """AudioPacket 필드가 올바른 값을 가지는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path, duration_sec=2.0, sample_rate=48000)
    config = _make_config(tmp_path, str(wav_path), chunk_size_ms=100, playback_speed=10.0)
    capture = FileMockCapture(config)

    await capture.start()
    try:
        packet = await asyncio.wait_for(
            capture.get_audio_queue().get(), timeout=2.0
        )
    finally:
        await capture.stop()

    # 필드 검증
    assert packet.packet_id == 0
    assert packet.sample_rate == 48000
    assert packet.bit_depth == 16
    assert packet.channels == 2  # 스테레오 WAV
    assert isinstance(packet.data, bytes)
    assert len(packet.data) > 0
    assert packet.timestamp_ns > 0


@pytest.mark.asyncio
async def test_audio_packet_ids_sequential(tmp_path):
    """AudioPacket의 packet_id가 순차적으로 증가하는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path, duration_sec=3.0)
    config = _make_config(tmp_path, str(wav_path), chunk_size_ms=100, playback_speed=10.0)
    capture = FileMockCapture(config)

    await capture.start()
    packets = []
    try:
        for _ in range(5):
            packet = await asyncio.wait_for(
                capture.get_audio_queue().get(), timeout=2.0
            )
            packets.append(packet)
    finally:
        await capture.stop()

    ids = [p.packet_id for p in packets]
    assert ids == list(range(5)), f"packet_id가 순차적이지 않습니다: {ids}"


@pytest.mark.asyncio
async def test_audio_packet_chunk_size(tmp_path):
    """AudioPacket의 data 크기가 chunk_size_ms에 비례하는지 확인합니다."""
    sample_rate = 48000
    chunk_size_ms = 100
    # 100ms × 48000Hz × 2채널 × 2바이트(int16) = 19200 bytes
    expected_bytes = int(sample_rate * chunk_size_ms / 1000) * 2 * 2  # 2ch × 2bytes

    wav_path = _make_wav_file(tmp_path, duration_sec=3.0, sample_rate=sample_rate)
    config = _make_config(tmp_path, str(wav_path), chunk_size_ms=chunk_size_ms, playback_speed=10.0)
    capture = FileMockCapture(config)

    await capture.start()
    try:
        packet = await asyncio.wait_for(
            capture.get_audio_queue().get(), timeout=2.0
        )
    finally:
        await capture.stop()

    assert len(packet.data) == expected_bytes, (
        f"청크 크기 불일치: 예상={expected_bytes}, 실제={len(packet.data)}"
    )


@pytest.mark.asyncio
async def test_audio_packet_timing_interval(tmp_path):
    """
    연속 AudioPacket 간 수신 간격이 chunk_size_ms ±허용범위인지 확인합니다.

    10배속으로 실행하므로 예상 간격은 chunk_size_ms / playback_speed = 10ms.
    허용 범위: ±5ms (asyncio 스케줄링 지연 고려)
    """
    chunk_size_ms = 100
    playback_speed = 10.0
    expected_interval_ms = chunk_size_ms / playback_speed  # 10ms
    tolerance_ms = 5.0

    wav_path = _make_wav_file(tmp_path, duration_sec=3.0)
    config = _make_config(
        tmp_path, str(wav_path),
        chunk_size_ms=chunk_size_ms,
        playback_speed=playback_speed,
    )
    capture = FileMockCapture(config)

    await capture.start()
    receive_times = []
    try:
        for _ in range(6):
            receive_times.append(time.monotonic())
            await asyncio.wait_for(
                capture.get_audio_queue().get(), timeout=2.0
            )
    finally:
        await capture.stop()

    # 연속 패킷 간 간격 계산 (첫 번째 제외 - 초기 지연 있을 수 있음)
    intervals_ms = [
        (receive_times[i + 1] - receive_times[i]) * 1000
        for i in range(1, len(receive_times) - 1)
    ]

    for i, interval_ms in enumerate(intervals_ms):
        assert abs(interval_ms - expected_interval_ms) <= tolerance_ms, (
            f"패킷 간격 범위 초과 (인덱스 {i+1}): "
            f"간격={interval_ms:.2f}ms, "
            f"예상={expected_interval_ms}ms ±{tolerance_ms}ms"
        )


# =============================================================================
# 비디오 프레임 생성 테스트
# =============================================================================

@pytest.mark.asyncio
async def test_video_frames_generated(tmp_path):
    """VideoFrame이 큐에 정상 생성되는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path, duration_sec=2.0)
    config = _make_config(tmp_path, str(wav_path), playback_speed=10.0)
    capture = FileMockCapture(config)

    await capture.start()
    try:
        frame = await asyncio.wait_for(
            capture.get_video_queue().get(), timeout=2.0
        )
    finally:
        await capture.stop()

    assert isinstance(frame, VideoFrame)


@pytest.mark.asyncio
async def test_video_frame_fields_valid(tmp_path):
    """VideoFrame 필드가 올바른 값을 가지는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path, duration_sec=2.0)
    config = _make_config(tmp_path, str(wav_path), playback_speed=10.0)
    capture = FileMockCapture(config)

    await capture.start()
    try:
        frame = await asyncio.wait_for(
            capture.get_video_queue().get(), timeout=2.0
        )
    finally:
        await capture.stop()

    assert frame.frame_id == 0
    assert frame.width == 1920
    assert frame.height == 1080
    assert frame.pixel_format == "yuv422"
    assert isinstance(frame.data, bytes)
    # YUV422: 1920 × 1080 × 2바이트 = 4,147,200 bytes
    assert len(frame.data) == 1920 * 1080 * 2
    assert frame.timestamp_ns > 0


# =============================================================================
# 종료 및 루프 테스트
# =============================================================================

@pytest.mark.asyncio
async def test_stop_terminates_cleanly(tmp_path):
    """stop() 호출 후 태스크가 정상 종료되는지 확인합니다."""
    wav_path = _make_wav_file(tmp_path, duration_sec=60.0)  # 긴 파일
    config = _make_config(tmp_path, str(wav_path), playback_speed=1.0)
    capture = FileMockCapture(config)

    await capture.start()
    await asyncio.sleep(0.05)  # 잠시 실행
    await capture.stop()

    assert not capture._running
    assert capture._audio_task is None
    assert capture._video_task is None


@pytest.mark.asyncio
async def test_loop_generates_more_packets_than_file(tmp_path):
    """loop=True 설정 시 파일 길이를 초과하는 패킷이 생성되는지 확인합니다."""
    # 0.5초 파일 → 5개 패킷만 생성 가능 (chunk=100ms)
    # loop=True + 10배속으로 총 15개 패킷 수집
    wav_path = _make_wav_file(tmp_path, duration_sec=0.5)
    config = _make_config(
        tmp_path, str(wav_path),
        chunk_size_ms=100,
        loop=True,
        playback_speed=10.0,
    )
    capture = FileMockCapture(config)

    await capture.start()
    packets = []
    try:
        for _ in range(15):
            packet = await asyncio.wait_for(
                capture.get_audio_queue().get(), timeout=5.0
            )
            packets.append(packet)
    finally:
        await capture.stop()

    # 파일 길이(0.5초/100ms = 5패킷)를 초과하여 15개가 생성되어야 함
    assert len(packets) == 15, f"루프 동작 실패: {len(packets)}개 패킷 (15개 기대)"


@pytest.mark.asyncio
async def test_file_not_found_stops_producer(tmp_path):
    """존재하지 않는 파일 경로 지정 시 프로듀서가 안전하게 중지되는지 확인합니다."""
    config = _make_config(tmp_path, "/nonexistent/path/audio.wav")
    capture = FileMockCapture(config)

    await capture.start()
    # 프로듀서가 파일 없음을 감지하고 _running을 False로 설정할 때까지 대기
    await asyncio.sleep(0.3)

    assert not capture._running
    await capture.stop()  # 이미 중지되었어도 안전하게 호출 가능
