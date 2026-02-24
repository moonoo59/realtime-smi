"""
Clova Speech gRPC 실시간 STT 스트리밍 클라이언트 모듈입니다.

역할:
- PCMChunk를 Clova Speech gRPC API에 실시간 스트리밍 전송
- partial/final STT 결과를 수신하여 result_queue에 전달
- 연결 끊김 시 exponential backoff 자동 재연결 (최대 5회)
- 각 단계 타임스탬프를 기록하여 지연 측정 지원

사용 예시:
    >>> streamer = ClovaSpeechStreamer(config)
    >>> await streamer.connect()
    >>> await streamer.stream_audio(pcm_chunk)
    >>> result = await streamer.get_result_queue().get()
    >>> await streamer.disconnect()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

import grpc
import grpc.aio

from src.audio import PCMChunk
from src.config.schema import AppConfig
from src.stt import STTResult, WordTiming
from src.stt import clova_speech_pb2 as pb2
from src.stt import clova_speech_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

# gRPC 연결 상태
_CONNECTED = "connected"
_DISCONNECTED = "disconnected"
_RECONNECTING = "reconnecting"


class ClovaSpeechStreamer:
    """
    Clova Speech gRPC 실시간 STT 스트리밍 클라이언트입니다.

    양방향 스트리밍(bidirectional streaming) RPC를 통해
    오디오를 전송하고 STT 결과를 수신합니다.

    재연결 전략:
        연결 끊김 감지 → exponential backoff 대기 → 재연결 시도
        1초 → 2초 → 4초 → 8초 → 16초 (최대 5회)
        5회 실패 시 result_queue에 None을 put하여 소비자에게 종료 신호 전달
    """

    def __init__(self, config: AppConfig) -> None:
        """
        ClovaSpeechStreamer를 초기화합니다.

        파라미터:
            config (AppConfig): 전체 애플리케이션 설정 객체
        """
        self._config = config
        self._stt_cfg = config.stt

        # gRPC 채널 및 스텁
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[pb2_grpc.ClovaSpeechServiceStub] = None

        # 스트리밍 call 객체
        self._stream_call = None

        # 오디오 전송용 큐 (스트림 요청 생성기가 소비)
        self._send_queue: asyncio.Queue[Optional[PCMChunk]] = asyncio.Queue(maxsize=200)

        # STT 결과 수신 큐
        self._result_queue: asyncio.Queue[Optional[STTResult]] = asyncio.Queue(
            maxsize=self._stt_cfg.result_queue_size
        )

        # 내부 상태
        self._connection_status: str = _DISCONNECTED
        self._result_id: int = 0
        self._reconnect_attempts: int = 0
        self._last_send_timestamp_ns: int = 0
        self._last_chunk_id: int = 0

        # 수신 루프 태스크
        self._receive_task: Optional[asyncio.Task] = None

        logger.info(
            f"ClovaSpeechStreamer 초기화: "
            f"endpoint={self._stt_cfg.endpoint}, "
            f"language={self._stt_cfg.language}"
        )

    # =========================================================================
    # 공개 인터페이스
    # =========================================================================

    def get_result_queue(self) -> asyncio.Queue[Optional[STTResult]]:
        """STTResult가 담기는 asyncio.Queue를 반환합니다. None은 종료 신호입니다."""
        return self._result_queue

    async def connect(self) -> None:
        """
        gRPC 채널을 생성하고 스트리밍 세션을 시작합니다.

        에러:
            grpc.aio.AioRpcError: gRPC 연결 실패 시
        """
        logger.info(f"gRPC 채널 연결 시작: {self._stt_cfg.endpoint}")

        self._channel = self._create_channel()
        self._stub = pb2_grpc.ClovaSpeechServiceStub(self._channel)

        await self._start_stream()

        self._connection_status = _CONNECTED
        self._reconnect_attempts = 0

        logger.info("gRPC 채널 연결 완료")

    async def stream_audio(self, chunk: PCMChunk) -> None:
        """
        PCM 청크를 gRPC 스트림으로 전송합니다.

        내부 send_queue에 청크를 추가하며, 실제 전송은
        스트림 요청 생성기(_request_generator)가 담당합니다.

        파라미터:
            chunk (PCMChunk): 전송할 PCM 청크
        """
        self._last_send_timestamp_ns = time.time_ns()
        self._last_chunk_id = chunk.chunk_id

        try:
            self._send_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # 가장 오래된 청크 제거 후 재삽입
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            await self._send_queue.put(chunk)
            logger.warning(f"전송 큐 오버플로우: chunk_id={chunk.chunk_id}")

        logger.debug(f"오디오 청크 전송 큐 추가: chunk_id={chunk.chunk_id}")

    async def disconnect(self) -> None:
        """
        스트림과 gRPC 채널을 종료합니다.

        수신 루프 태스크를 취소하고 채널을 닫습니다.
        result_queue에 None을 put하여 소비자에게 종료를 알립니다.
        """
        logger.info("ClovaSpeechStreamer 연결 종료 시작")
        self._connection_status = _DISCONNECTED

        # 전송 큐에 종료 신호
        await self._send_queue.put(None)

        # 수신 루프 태스크 취소
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # gRPC 채널 닫기
        if self._channel:
            await self._channel.close()
            self._channel = None

        # 소비자에게 종료 신호
        await self._result_queue.put(None)

        logger.info("ClovaSpeechStreamer 연결 종료 완료")

    # =========================================================================
    # 내부 스트리밍 메서드
    # =========================================================================

    def _create_channel(self) -> grpc.aio.Channel:
        """
        인증 설정에 따라 SSL 또는 일반 gRPC 채널을 생성합니다.

        반환값:
            grpc.aio.Channel: gRPC 비동기 채널
        """
        metadata = self._build_metadata()

        if self._stt_cfg.use_ssl:
            credentials = grpc.ssl_channel_credentials()
            return grpc.aio.secure_channel(
                self._stt_cfg.endpoint,
                credentials,
                options=[("grpc.keepalive_time_ms", 30000)],
                interceptors=[_MetadataInterceptor(metadata)],
            )
        else:
            return grpc.aio.insecure_channel(
                self._stt_cfg.endpoint,
                options=[("grpc.keepalive_time_ms", 30000)],
                interceptors=[_MetadataInterceptor(metadata)],
            )

    def _build_metadata(self) -> list[tuple[str, str]]:
        """
        gRPC 요청에 첨부할 인증 메타데이터를 생성합니다.

        반환값:
            list[tuple[str, str]]: gRPC 메타데이터 목록
        """
        metadata = []
        if self._stt_cfg.api_key:
            metadata.append(("x-clovaspeech-api-gw-api-key", self._stt_cfg.api_key))
        if self._stt_cfg.secret_key:
            metadata.append(("x-clovaspeech-api-gw-secret-key", self._stt_cfg.secret_key))
        return metadata

    async def _start_stream(self) -> None:
        """
        양방향 스트리밍 RPC를 시작하고 수신 루프 태스크를 생성합니다.
        """
        self._stream_call = self._stub.Recognize(
            self._request_generator(),
            timeout=None,  # 무제한 (장시간 방송용)
        )

        self._receive_task = asyncio.create_task(
            self._receive_loop(), name="stt_receive_loop"
        )

        logger.info("gRPC 스트리밍 세션 시작")

    async def _request_generator(self) -> AsyncGenerator:
        """
        send_queue에서 PCMChunk를 꺼내 gRPC 요청으로 변환하는 비동기 생성기입니다.

        최초에는 설정(RecognizeConfig)을 전송하고,
        이후에는 오디오 데이터(AudioContent)를 전송합니다.
        None을 받으면 스트림을 종료합니다.
        """
        # 최초 요청: 세션 설정
        config_request = pb2.RecognizeRequest(
            config=pb2.RecognizeConfig(
                language_code=self._stt_cfg.language,
                encoding=pb2.LINEAR16,
                sample_rate_hertz=16000,
                domain=self._stt_cfg.domain,
                model=self._stt_cfg.model,
                enable_partial=True,
            )
        )
        yield config_request
        logger.debug("STT 세션 설정 전송 완료")

        # 이후 요청: 오디오 청크
        while True:
            chunk = await self._send_queue.get()

            # None은 종료 신호
            if chunk is None:
                logger.debug("전송 생성기 종료 신호 수신")
                return

            audio_request = pb2.RecognizeRequest(
                audio_content=pb2.AudioContent(content=chunk.data)
            )
            yield audio_request

    async def _receive_loop(self) -> None:
        """
        gRPC 스트림에서 STT 결과를 수신하는 루프입니다.

        수신된 결과를 STTResult로 변환하여 result_queue에 추가합니다.
        연결이 끊기면 자동 재연결을 시도합니다.
        """
        logger.info("STT 수신 루프 시작")

        try:
            async for response in self._stream_call:
                receive_ts = time.time_ns()

                result = self._parse_response(response, receive_ts)
                if result is None:
                    continue

                await self._result_queue.put(result)

                logger.debug(
                    f"STT 결과 수신: type={result.type}, "
                    f"text='{result.text[:30]}...', "
                    f"latency={int((receive_ts - result.send_timestamp_ns) / 1_000_000)}ms"
                )

        except asyncio.CancelledError:
            logger.info("STT 수신 루프 취소됨")
            raise

        except grpc.aio.AioRpcError as rpc_error:
            if self._connection_status == _DISCONNECTED:
                # 의도적 종료
                return
            logger.error(
                f"gRPC 스트림 오류: code={rpc_error.code()}, "
                f"details={rpc_error.details()}"
            )
            await self._reconnect_with_backoff()

        except StopAsyncIteration:
            logger.info("gRPC 스트림 종료 (서버측 완료)")
            if self._connection_status != _DISCONNECTED:
                await self._reconnect_with_backoff()

        except Exception as exc:
            logger.error(f"STT 수신 루프 예상치 못한 오류: {exc}", exc_info=True)
            if self._connection_status != _DISCONNECTED:
                await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self) -> None:
        """
        Exponential backoff를 적용하여 gRPC 재연결을 시도합니다.

        재연결 대기 시간: base_sec × 2^attempts (최대 max_sec)
        최대 max_reconnect_attempts회 시도 후 포기합니다.
        """
        self._connection_status = _RECONNECTING
        base_sec = self._stt_cfg.reconnect_backoff_base_sec
        max_sec = self._stt_cfg.reconnect_backoff_max_sec
        max_attempts = self._stt_cfg.max_reconnect_attempts

        while self._reconnect_attempts < max_attempts:
            wait_sec = min(base_sec * (2 ** self._reconnect_attempts), max_sec)
            self._reconnect_attempts += 1

            logger.warning(
                f"gRPC 재연결 시도 {self._reconnect_attempts}/{max_attempts}: "
                f"{wait_sec}초 대기"
            )
            await asyncio.sleep(wait_sec)

            try:
                # 기존 채널 닫기
                if self._channel:
                    await self._channel.close()

                # 새 채널/스트림 생성
                self._channel = self._create_channel()
                self._stub = pb2_grpc.ClovaSpeechServiceStub(self._channel)

                # 전송 큐 비우기 (오래된 데이터 제거)
                while not self._send_queue.empty():
                    try:
                        self._send_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                await self._start_stream()
                self._connection_status = _CONNECTED
                self._reconnect_attempts = 0

                logger.info(f"gRPC 재연결 성공 ({self._reconnect_attempts}번째 시도)")
                return

            except Exception as exc:
                logger.error(f"재연결 시도 {self._reconnect_attempts} 실패: {exc}")

        # 모든 재연결 시도 실패
        logger.error(
            f"gRPC 재연결 {max_attempts}회 모두 실패. STT 스트리밍 중단."
        )
        self._connection_status = _DISCONNECTED
        await self._result_queue.put(None)  # 소비자에게 종료 신호

    def _parse_response(
        self, response: pb2.RecognizeResponse, receive_ts: int
    ) -> Optional[STTResult]:
        """
        gRPC RecognizeResponse를 STTResult로 변환합니다.

        파라미터:
            response: gRPC 응답 메시지
            receive_ts: 수신 시각 (nanoseconds)

        반환값:
            Optional[STTResult]: 변환된 결과 (텍스트가 없으면 None)
        """
        if not response.text:
            return None

        result_type = "partial" if response.result_type == pb2.PARTIAL else "final"

        words = [
            WordTiming(
                word=w.word,
                start_ms=w.start_ms,
                end_ms=w.end_ms,
            )
            for w in response.words
        ]

        result_id = self._result_id
        self._result_id += 1

        return STTResult(
            result_id=result_id,
            type=result_type,
            text=response.text,
            send_timestamp_ns=self._last_send_timestamp_ns,
            receive_timestamp_ns=receive_ts,
            confidence=response.confidence,
            words=words,
            last_chunk_id=self._last_chunk_id,
        )


# =============================================================================
# gRPC 인증 인터셉터
# =============================================================================

class _MetadataInterceptor(grpc.aio.UnaryUnaryClientInterceptor,
                            grpc.aio.StreamStreamClientInterceptor,
                            grpc.aio.UnaryStreamClientInterceptor,
                            grpc.aio.StreamUnaryClientInterceptor):
    """
    모든 gRPC 요청에 인증 메타데이터를 자동으로 첨부하는 인터셉터입니다.
    """

    def __init__(self, metadata: list[tuple[str, str]]) -> None:
        self._metadata = metadata

    def _add_metadata(self, client_call_details):
        """기존 메타데이터에 인증 정보를 추가합니다."""
        if not self._metadata:
            return client_call_details

        existing = list(client_call_details.metadata or [])
        new_details = grpc.aio.ClientCallDetails()
        new_details.method = client_call_details.method
        new_details.timeout = client_call_details.timeout
        new_details.metadata = existing + self._metadata
        new_details.credentials = client_call_details.credentials
        new_details.wait_for_ready = client_call_details.wait_for_ready
        return new_details

    async def intercept_unary_unary(self, continuation, client_call_details, request):
        return await continuation(self._add_metadata(client_call_details), request)

    async def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return await continuation(self._add_metadata(client_call_details), request_iterator)

    async def intercept_unary_stream(self, continuation, client_call_details, request):
        return await continuation(self._add_metadata(client_call_details), request)

    async def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        return await continuation(self._add_metadata(client_call_details), request_iterator)
