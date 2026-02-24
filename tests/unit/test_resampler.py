"""
AudioResampler 단위 테스트

검증 항목:
- 리샘플링 출력 샘플레이트가 16000Hz인지 확인
- 24bit → 16bit 변환 정확성
- 스테레오 → 모노 믹스다운
- 게인 조정 (dB)
- RMS/Peak 레벨 계산
- chunk_size_ms 단위 분할
- VAD: 무음 구간 False, 사인파(비음성) 처리
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from src.audio import PCMChunk
from src.audio.resampler import (
    AudioResampler,
    _bytes_to_numpy,
    _db_to_linear,
    _decode_24bit,
    _mixdown_to_mono,
    _normalize_to_float32,
    _resample,
)
from src.capture import AudioPacket
from src.config.schema import AppConfig


# =============================================================================
# 테스트 헬퍼
# =============================================================================

def _make_config(
    chunk_size_ms: int = 100,
    gain_db: float = 0.0,
    vad_enabled: bool = False,
    out_sample_rate: int = 16000,
) -> AppConfig:
    """테스트용 AppConfig를 생성합니다."""
    return AppConfig(**{
        "system": {"mode": "file"},
        "audio": {
            "output_sample_rate": out_sample_rate,
            "chunk_size_ms": chunk_size_ms,
            "gain_db": gain_db,
            "vad": {"enabled": vad_enabled, "mode": 3, "padding_ms": 0},
        },
    })


def _make_audio_packet(
    data: bytes,
    sample_rate: int = 48000,
    bit_depth: int = 16,
    channels: int = 1,
    packet_id: int = 0,
) -> AudioPacket:
    """테스트용 AudioPacket을 생성합니다."""
    return AudioPacket(
        packet_id=packet_id,
        timestamp_ns=1_000_000_000,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=channels,
        data=data,
    )


def _make_sine_wave(
    freq_hz: float,
    duration_sec: float,
    sample_rate: int,
    amplitude: float = 0.5,
    dtype=np.int16,
) -> np.ndarray:
    """사인파 PCM 데이터를 생성합니다."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    wave = np.sin(2 * np.pi * freq_hz * t) * amplitude
    max_val = 2 ** (np.dtype(dtype).itemsize * 8 - 1) - 1
    return (wave * max_val).astype(dtype)


def _make_silence(num_samples: int, dtype=np.int16) -> np.ndarray:
    """무음 PCM 데이터를 생성합니다."""
    return np.zeros(num_samples, dtype=dtype)


# =============================================================================
# 리샘플링 테스트
# =============================================================================

def test_resample_48k_to_16k():
    """48kHz → 16kHz 리샘플링 출력 길이가 정확한지 확인합니다."""
    from_rate = 48000
    to_rate = 16000
    duration_sec = 0.1  # 100ms

    samples_in = _make_sine_wave(1000, duration_sec, from_rate).astype(np.float32) / 32767.0
    samples_out = _resample(samples_in, from_rate, to_rate)

    # 출력 샘플 수 = 입력 샘플 수 × (to_rate / from_rate)
    expected_samples = int(len(samples_in) * to_rate / from_rate)
    # scipy resample_poly는 반올림으로 인해 ±1 오차 허용
    assert abs(len(samples_out) - expected_samples) <= 1, (
        f"리샘플링 출력 길이 불일치: {len(samples_out)} vs {expected_samples}"
    )


def test_resample_output_sample_rate_via_resampler():
    """AudioResampler를 통한 리샘플링 후 PCMChunk sample_rate가 16000Hz인지 확인합니다."""
    config = _make_config(chunk_size_ms=100)
    resampler = AudioResampler(config)

    # 48kHz/16bit/mono, 0.5초 사인파
    sine = _make_sine_wave(1000, 0.5, 48000)
    packet = _make_audio_packet(sine.tobytes(), sample_rate=48000, bit_depth=16, channels=1)

    chunks = resampler.resample(packet)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.sample_rate == 16000, f"sample_rate={chunk.sample_rate} (16000 기대)"


def test_resample_preserves_audio_content():
    """리샘플링 후 신호 에너지가 크게 손실되지 않는지 확인합니다."""
    config = _make_config(chunk_size_ms=200)
    resampler = AudioResampler(config)

    # 1초 사인파
    sine_48k = _make_sine_wave(440, 1.0, 48000, amplitude=0.8)
    packet = _make_audio_packet(sine_48k.tobytes(), sample_rate=48000, bit_depth=16, channels=1)

    chunks = resampler.resample(packet)

    assert len(chunks) > 0
    # 출력 RMS가 0이 아닌지 (신호가 살아있는지) 확인
    total_rms = sum(c.rms for c in chunks) / len(chunks)
    assert total_rms > 0.1, f"리샘플링 후 RMS가 너무 낮습니다: {total_rms}"


# =============================================================================
# 비트뎁스 변환 테스트
# =============================================================================

def test_decode_16bit_to_numpy():
    """16bit PCM bytes가 올바른 numpy 배열로 변환되는지 확인합니다."""
    samples = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    arr = _bytes_to_numpy(samples.tobytes(), bit_depth=16, channels=1)
    np.testing.assert_array_equal(arr, samples)


def test_decode_24bit():
    """24bit PCM이 올바르게 디코딩되는지 확인합니다."""
    # 알려진 24bit 값으로 테스트
    # 양수 값: 0x000100 = 256
    # 음수 값: 0xFF0000 = -65536 (24bit 2의 보수)
    raw_positive = bytes([0x00, 0x01, 0x00])  # 256 in little-endian 24bit
    raw_negative = bytes([0x00, 0x00, 0xFF])  # -65536 in little-endian 24bit

    pos_arr = _decode_24bit(raw_positive)
    neg_arr = _decode_24bit(raw_negative)

    assert pos_arr[0] == 256, f"24bit 양수 디코딩 실패: {pos_arr[0]}"
    assert neg_arr[0] < 0, f"24bit 음수 디코딩 실패: {neg_arr[0]}"


def test_resample_24bit_input():
    """24bit 입력이 올바르게 처리되는지 확인합니다."""
    config = _make_config(chunk_size_ms=100)
    resampler = AudioResampler(config)

    # 24bit 사인파 생성 (int32로 저장 후 24bit로 변환)
    duration_sec = 0.5
    sample_rate = 48000
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sine_int32 = (np.sin(2 * np.pi * 1000 * t) * (2**23 - 1)).astype(np.int32)

    # 24bit로 변환 (하위 3바이트만 사용, little-endian)
    sine_24bit = b""
    for s in sine_int32:
        # little-endian 3바이트
        sine_24bit += struct.pack("<i", s)[:3]

    packet = _make_audio_packet(sine_24bit, sample_rate=48000, bit_depth=24, channels=1)
    chunks = resampler.resample(packet)

    assert len(chunks) > 0


# =============================================================================
# 스테레오→모노 믹스다운 테스트
# =============================================================================

def test_stereo_to_mono_mixdown():
    """스테레오 배열이 모노로 올바르게 믹스다운되는지 확인합니다."""
    # L채널=1.0, R채널=0.0 → 모노 평균=0.5
    stereo = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    mono = _mixdown_to_mono(stereo, channels=2)

    assert mono.shape == (3,)
    np.testing.assert_allclose(mono, [0.5, 0.5, 0.5], atol=1e-6)


def test_stereo_packet_becomes_mono_chunk():
    """스테레오 AudioPacket이 모노 PCMChunk로 변환되는지 확인합니다."""
    config = _make_config(chunk_size_ms=100)
    resampler = AudioResampler(config)

    # 48kHz/16bit/stereo, 0.5초
    sine = _make_sine_wave(1000, 0.5, 48000)
    stereo = np.column_stack([sine, sine])  # L=R
    packet = _make_audio_packet(stereo.tobytes(), sample_rate=48000, bit_depth=16, channels=2)

    chunks = resampler.resample(packet)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.channels == 1, f"모노 변환 실패: channels={chunk.channels}"


# =============================================================================
# 게인 조정 테스트
# =============================================================================

def test_apply_gain_positive_db():
    """양수 게인(+6dB)이 올바르게 적용되는지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    samples = np.array([1000, 2000, 3000], dtype=np.int16)
    gained = resampler.apply_gain(samples.tobytes(), gain_db=6.0)
    result = np.frombuffer(gained, dtype=np.int16)

    # +6dB ≈ ×2.0
    expected = np.clip(np.array([1000, 2000, 3000]) * _db_to_linear(6.0), -32767, 32767)
    np.testing.assert_allclose(result, expected, atol=2.0)  # int16 반올림 허용


def test_apply_gain_zero_db():
    """0dB 게인은 신호를 변경하지 않아야 합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    samples = np.array([1000, -1000, 2000], dtype=np.int16)
    gained = resampler.apply_gain(samples.tobytes(), gain_db=0.0)
    result = np.frombuffer(gained, dtype=np.int16)

    np.testing.assert_array_equal(result, samples)


def test_apply_gain_clips_at_max():
    """게인 적용 시 int16 최대값을 초과하지 않는지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    # 풀스케일 신호에 +20dB 적용 → 클리핑 발생해야 함
    samples = np.array([32767, -32767], dtype=np.int16)
    gained = resampler.apply_gain(samples.tobytes(), gain_db=20.0)
    result = np.frombuffer(gained, dtype=np.int16)

    assert result[0] <= 32767
    assert result[1] >= -32767


# =============================================================================
# RMS/Peak 레벨 테스트
# =============================================================================

def test_calculate_rms_silence():
    """무음 신호의 RMS가 0.0인지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    silence = np.zeros(1600, dtype=np.int16)
    rms = resampler.calculate_rms(silence.tobytes())

    assert rms == 0.0


def test_calculate_rms_full_scale():
    """풀스케일 사인파의 RMS가 약 0.707(1/√2)인지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    # 풀스케일 사인파 RMS = 1/√2 ≈ 0.707
    sine = _make_sine_wave(1000, 0.1, 16000, amplitude=1.0)
    rms = resampler.calculate_rms(sine.tobytes())

    assert abs(rms - 1.0 / math.sqrt(2)) < 0.02, f"RMS 오차: {rms}"


def test_calculate_peak_silence():
    """무음 신호의 Peak이 0.0인지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    silence = np.zeros(1600, dtype=np.int16)
    peak = resampler.calculate_peak(silence.tobytes())

    assert peak == 0.0


def test_calculate_peak_known_value():
    """알려진 값의 Peak가 정확한지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    # 최대값 16383 (= 32767 / 2, peak ≈ 0.5)
    samples = np.array([0, 16383, -16383, 0], dtype=np.int16)
    peak = resampler.calculate_peak(samples.tobytes())

    assert abs(peak - 16383 / 32767) < 0.001


# =============================================================================
# 청크 분할 테스트
# =============================================================================

def test_split_chunks_even_division():
    """정확히 나누어지는 경우 청크 크기가 정확한지 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    # 1600 샘플 × 2바이트 = 3200 bytes → 100ms at 16kHz
    data = bytes(3200)
    chunks = resampler.split_chunks(data, chunk_size=800)

    assert len(chunks) == 4
    assert all(len(c) == 800 for c in chunks)


def test_split_chunks_with_remainder():
    """나머지가 있는 경우 마지막 청크 크기 확인합니다."""
    config = _make_config()
    resampler = AudioResampler(config)

    data = bytes(1000)
    chunks = resampler.split_chunks(data, chunk_size=300)

    assert len(chunks) == 4  # 300+300+300+100
    assert len(chunks[-1]) == 100  # 나머지


# =============================================================================
# VAD 통합 테스트
# =============================================================================

def test_vad_disabled_passes_all_chunks():
    """VAD 비활성화 시 무음 구간도 통과하는지 확인합니다."""
    config = _make_config(chunk_size_ms=100, vad_enabled=False)
    resampler = AudioResampler(config)

    # 완전한 무음 0.5초
    silence = np.zeros(int(48000 * 0.5), dtype=np.int16)
    packet = _make_audio_packet(silence.tobytes(), sample_rate=48000, bit_depth=16, channels=1)

    chunks = resampler.resample(packet)

    # VAD 비활성화이므로 무음도 통과해야 함
    assert len(chunks) > 0


def test_vad_silence_blocked(tmp_path):
    """VAD 활성화 시 완전한 무음이 필터링되는지 확인합니다."""
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        pytest.skip("webrtcvad가 설치되지 않아 VAD 테스트를 건너뜁니다")

    config = _make_config(chunk_size_ms=30, vad_enabled=True)
    resampler = AudioResampler(config)

    # 완전한 무음 (VAD가 비음성으로 판별해야 함)
    silence = np.zeros(int(48000 * 0.5), dtype=np.int16)
    packet = _make_audio_packet(silence.tobytes(), sample_rate=48000, bit_depth=16, channels=1)

    chunks = resampler.resample(packet)

    assert len(chunks) == 0, f"무음이 VAD를 통과함: {len(chunks)}개 청크"


# =============================================================================
# 잔여 버퍼 테스트
# =============================================================================

def test_residual_buffer_accumulation():
    """잔여 샘플이 다음 패킷과 합쳐져 청크를 완성하는지 확인합니다."""
    config = _make_config(chunk_size_ms=100, vad_enabled=False)
    resampler = AudioResampler(config)

    # 16kHz 기준 chunk = 1600 샘플
    # 각 패킷이 1000 샘플만 생성하도록 (리샘플 후)
    # 48kHz 3000 샘플 → 리샘플 → 16kHz 1000 샘플
    n_48k_samples = 3000  # → 16kHz에서 1000 샘플
    sine = _make_sine_wave(1000, n_48k_samples / 48000, 48000)

    packet1 = _make_audio_packet(sine.tobytes(), sample_rate=48000, bit_depth=16, channels=1, packet_id=0)
    packet2 = _make_audio_packet(sine.tobytes(), sample_rate=48000, bit_depth=16, channels=1, packet_id=1)

    chunks1 = resampler.resample(packet1)  # 1000 샘플 → 청크 없음 (1600 미만)
    chunks2 = resampler.resample(packet2)  # 1000+1000=2000 샘플 → 1청크 + 잔여400

    # 두 패킷을 합쳐야 최소 1청크가 완성됨
    total_chunks = len(chunks1) + len(chunks2)
    assert total_chunks >= 1, "잔여 버퍼 축적 실패: 2패킷 합쳐도 청크 생성 안됨"


# =============================================================================
# 핫스왑 설정 업데이트 테스트
# =============================================================================

def test_update_config_changes_gain():
    """update_config 호출 후 게인이 즉시 변경되는지 확인합니다."""
    config_before = _make_config(gain_db=0.0)
    config_after = _make_config(gain_db=6.0)
    resampler = AudioResampler(config_before)

    gain_before = resampler._gain_linear
    resampler.update_config(config_after)
    gain_after = resampler._gain_linear

    assert abs(gain_after - gain_before) > 0.01, "gain_linear가 변경되지 않았습니다"
    assert gain_after > gain_before, "+6dB 적용 후 gain_linear가 증가해야 합니다"


def test_update_config_creates_new_vad_instance():
    """update_config 호출 후 VadFilter가 새 인스턴스로 교체되는지 확인합니다."""
    config_before = _make_config(vad_enabled=False)
    config_after = _make_config(vad_enabled=False, chunk_size_ms=50)
    resampler = AudioResampler(config_before)

    vad_before = resampler._vad
    resampler.update_config(config_after)

    assert resampler._vad is not vad_before, "update_config 후 새 VadFilter 인스턴스가 생성되어야 합니다"


def test_update_config_preserves_sample_rate():
    """update_config 호출 후 출력 샘플레이트가 유지되는지 확인합니다."""
    config = _make_config(out_sample_rate=16000)
    resampler = AudioResampler(config)

    new_config = _make_config(gain_db=3.0)
    resampler.update_config(new_config)

    assert resampler._out_sample_rate == 16000, "출력 샘플레이트가 변경되면 안됩니다"
