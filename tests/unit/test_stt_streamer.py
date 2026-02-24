"""
ClovaSpeechStreamer 단위 테스트

실제 Clova Speech API 없이 mock gRPC 서버/응답을 사용하여 테스트합니다.

검증 항목:
- get_result_queue() 인터페이스
- stream_audio() 큐 추가
- STTResult 파싱 (_parse_response)
- exponential backoff 재연결 로직
- disconnect() 후 종료 신호 전달
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.audio import PCMChunk
from src.config.schema import AppConfig
from src.stt import STTResult, WordTiming
from src.stt import clova_speech_pb2 as pb2
from src.stt.clova_streamer import ClovaSpeechStreamer


# =============================================================================
# 테스트 헬퍼
# =============================================================================

def _make_config(
    endpoint: str = "localhost:50051",
    api_key: str = "test-key",
    use_ssl: bool = False,
    max_reconnect_attempts: int = 3,
    reconnect_backoff_base_sec: int = 1,
) -> AppConfig:
    """테스트용 AppConfig를 생성합니다."""
    return AppConfig(**{
        "stt": {
            "endpoint": endpoint,
            "api_key": api_key,
            "use_ssl": use_ssl,
            "max_reconnect_attempts": max_reconnect_attempts,
            "reconnect_backoff_base_sec": reconnect_backoff_base_sec,
            "reconnect_backoff_max_sec": 16,
            "result_queue_size": 50,
            "language": "ko-KR",
            "domain": "general",
            "model": "general",
        },
    })


def _make_pcm_chunk(chunk_id: int = 0) -> PCMChunk:
    """테스트용 PCMChunk를 생성합니다."""
    return PCMChunk(
        chunk_id=chunk_id,
        capture_timestamp_ns=time.time_ns(),
        sample_rate=16000,
        bit_depth=16,
        channels=1,
        data=b"\x00" * 3200,  # 100ms silence at 16kHz/16bit
        rms=0.0,
        peak=0.0,
    )


def _make_grpc_response(
    text: str,
    result_type: int = pb2.FINAL,
    confidence: float = 0.95,
) -> pb2.RecognizeResponse:
    """테스트용 gRPC RecognizeResponse를 생성합니다."""
    return pb2.RecognizeResponse(
        result_type=result_type,
        text=text,
        confidence=confidence,
        words=[
            pb2.WordInfo(word=w, start_ms=i * 200, end_ms=(i + 1) * 200)
            for i, w in enumerate(text.split())
        ],
    )


# =============================================================================
# 인터페이스 테스트
# =============================================================================

def test_get_result_queue_returns_asyncio_queue():
    """get_result_queue()가 asyncio.Queue를 반환하는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    queue = streamer.get_result_queue()

    assert isinstance(queue, asyncio.Queue)


def test_get_result_queue_same_instance():
    """매번 동일한 큐 인스턴스를 반환하는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    assert streamer.get_result_queue() is streamer.get_result_queue()


# =============================================================================
# stream_audio 테스트
# =============================================================================

@pytest.mark.asyncio
async def test_stream_audio_adds_to_send_queue():
    """stream_audio()가 내부 send_queue에 청크를 추가하는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    chunk = _make_pcm_chunk(chunk_id=0)
    await streamer.stream_audio(chunk)

    assert not streamer._send_queue.empty()
    queued = streamer._send_queue.get_nowait()
    assert queued is chunk


@pytest.mark.asyncio
async def test_stream_audio_overflow_handling():
    """send_queue가 가득 찼을 때 오래된 항목을 제거하는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    # send_queue 최대 크기(200)를 초과하는 청크 생성
    for i in range(200):
        streamer._send_queue.put_nowait(_make_pcm_chunk(chunk_id=i))

    # 큐가 꽉 찬 상태에서 새 청크 추가
    new_chunk = _make_pcm_chunk(chunk_id=999)
    await streamer.stream_audio(new_chunk)

    # 큐 크기는 여전히 200 이하여야 함
    assert streamer._send_queue.qsize() <= 200


# =============================================================================
# STTResult 파싱 테스트
# =============================================================================

def test_parse_response_final():
    """FINAL 응답이 올바르게 STTResult로 변환되는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)
    streamer._last_send_timestamp_ns = 1_000_000_000

    response = _make_grpc_response("안녕하세요 반갑습니다", result_type=pb2.FINAL, confidence=0.95)
    receive_ts = 2_000_000_000

    result = streamer._parse_response(response, receive_ts)

    assert result is not None
    assert result.type == "final"
    assert result.text == "안녕하세요 반갑습니다"
    assert result.confidence == pytest.approx(0.95, abs=0.001)
    assert result.send_timestamp_ns == 1_000_000_000
    assert result.receive_timestamp_ns == 2_000_000_000


def test_parse_response_partial():
    """PARTIAL 응답이 올바르게 STTResult로 변환되는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)
    streamer._last_send_timestamp_ns = 500_000_000

    response = _make_grpc_response("안녕하", result_type=pb2.PARTIAL, confidence=0.7)
    result = streamer._parse_response(response, 600_000_000)

    assert result is not None
    assert result.type == "partial"
    assert result.text == "안녕하"


def test_parse_response_empty_text_returns_none():
    """텍스트가 비어있는 응답은 None을 반환하는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    response = pb2.RecognizeResponse(result_type=pb2.PARTIAL, text="")
    result = streamer._parse_response(response, time.time_ns())

    assert result is None


def test_parse_response_word_timings():
    """단어 타임스탬프가 올바르게 파싱되는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)
    streamer._last_send_timestamp_ns = 0

    response = pb2.RecognizeResponse(
        result_type=pb2.FINAL,
        text="테스트",
        confidence=0.9,
        words=[
            pb2.WordInfo(word="테스트", start_ms=0, end_ms=500),
        ],
    )
    result = streamer._parse_response(response, time.time_ns())

    assert result is not None
    assert len(result.words) == 1
    assert result.words[0].word == "테스트"
    assert result.words[0].start_ms == 0
    assert result.words[0].end_ms == 500


# =============================================================================
# 재연결 로직 테스트
# =============================================================================

@pytest.mark.asyncio
async def test_reconnect_backoff_increments_attempts():
    """재연결 시도마다 backoff 대기 시간이 증가하는지 확인합니다."""
    config = _make_config(
        max_reconnect_attempts=3,
        reconnect_backoff_base_sec=1,
    )
    streamer = ClovaSpeechStreamer(config)
    streamer._connection_status = "reconnecting"

    sleep_calls = []

    async def mock_sleep(sec):
        sleep_calls.append(sec)

    # 채널/스텁 생성 실패를 시뮬레이션
    with patch("src.stt.clova_streamer.asyncio.sleep", side_effect=mock_sleep), \
         patch.object(streamer, "_create_channel", side_effect=Exception("연결 실패")):
        await streamer._reconnect_with_backoff()

    # backoff 순서: 1초 → 2초 → 4초 (최대 3회)
    assert len(sleep_calls) == 3
    assert sleep_calls[0] == 1
    assert sleep_calls[1] == 2
    assert sleep_calls[2] == 4


@pytest.mark.asyncio
async def test_reconnect_sends_none_after_max_attempts():
    """최대 재연결 횟수 실패 후 result_queue에 None이 전달되는지 확인합니다."""
    config = _make_config(max_reconnect_attempts=2)
    streamer = ClovaSpeechStreamer(config)
    streamer._connection_status = "reconnecting"

    async def mock_sleep(sec):
        pass

    with patch("src.stt.clova_streamer.asyncio.sleep", side_effect=mock_sleep), \
         patch.object(streamer, "_create_channel", side_effect=Exception("연결 실패")):
        await streamer._reconnect_with_backoff()

    # result_queue에 None이 있어야 함 (종료 신호)
    result = streamer._result_queue.get_nowait()
    assert result is None


@pytest.mark.asyncio
async def test_reconnect_succeeds_on_second_attempt():
    """두 번째 시도에서 재연결 성공 시 정상 상태로 복귀하는지 확인합니다."""
    config = _make_config(max_reconnect_attempts=3)
    streamer = ClovaSpeechStreamer(config)
    streamer._connection_status = "reconnecting"
    streamer._stub = MagicMock()

    attempt_count = 0

    def side_effect_channel():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise Exception("첫 번째 실패")
        return MagicMock()

    async def mock_sleep(sec):
        pass

    async def mock_start_stream():
        pass

    with patch("src.stt.clova_streamer.asyncio.sleep", side_effect=mock_sleep), \
         patch.object(streamer, "_create_channel", side_effect=side_effect_channel), \
         patch.object(streamer, "_start_stream", side_effect=mock_start_stream):
        await streamer._reconnect_with_backoff()

    assert streamer._connection_status == "connected"
    assert streamer._reconnect_attempts == 0


# =============================================================================
# disconnect 테스트
# =============================================================================

@pytest.mark.asyncio
async def test_disconnect_sends_none_to_result_queue():
    """disconnect() 후 result_queue에 None(종료 신호)이 전달되는지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    # 채널 mock (실제 gRPC 연결 없이)
    mock_channel = AsyncMock()
    streamer._channel = mock_channel

    await streamer.disconnect()

    # result_queue에 None이 있어야 함
    result = streamer._result_queue.get_nowait()
    assert result is None


@pytest.mark.asyncio
async def test_disconnect_sets_status_disconnected():
    """disconnect() 후 연결 상태가 disconnected인지 확인합니다."""
    config = _make_config()
    streamer = ClovaSpeechStreamer(config)

    mock_channel = AsyncMock()
    streamer._channel = mock_channel

    await streamer.disconnect()

    assert streamer._connection_status == "disconnected"
