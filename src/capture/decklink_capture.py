"""
Blackmagic DeckLink SDK 기반 실제 SDI 캡처 모듈입니다.

역할:
- DeckLink SDK ctypes 바인딩을 통해 SDI 입력에서 비디오/오디오 캡처
- 비디오 콜백: SDI YUV422 프레임 → VideoFrame → video_queue
- 오디오 콜백: SDI PCM 패킷 → AudioPacket → audio_queue
- FileMockCapture와 동일한 asyncio.Queue 인터페이스 제공
- YUV422 픽셀 포맷 정합성: VideoCompositor._uyvy_to_bgr()과 연동

사용 예시:
    >>> capture = DeckLinkCapture(config)
    >>> await capture.start()
    >>> audio_queue = capture.get_audio_queue()
    >>> packet = await audio_queue.get()
    >>> await capture.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from src.capture import AudioPacket, VideoFrame
from src.capture.decklink_bindings import (
    BMDAudioSampleRate,
    BMDAudioSampleType,
    BMDPixelFormat,
    DeckLinkAPIError,
    DeckLinkAudioPacket,
    DeckLinkDevice,
    DeckLinkInput,
    DeckLinkInputCallback,
    DeckLinkSDKNotFoundError,
    DeckLinkVideoFrame,
    VIDEO_MODE_MAP,
    open_device,
)
from src.config.schema import AppConfig

logger = logging.getLogger(__name__)


class DeckLinkCapture:
    """
    Blackmagic DeckLink SDK를 통한 실제 SDI 캡처 클래스입니다.

    DeckLink SDK의 IDeckLinkInputCallback을 Python으로 구현하여
    비디오/오디오 프레임을 asyncio.Queue에 전달합니다.

    아키텍처:
        [DeckLink SDK 콜백 스레드]
            ↓ _on_frame_arrived() (동기)
        [asyncio loop.call_soon_threadsafe()]
            ↓ _enqueue_video() / _enqueue_audio() (비동기)
        [video_queue] / [audio_queue]
    """

    def __init__(self, config: AppConfig) -> None:
        """
        DeckLinkCapture를 초기화합니다.

        파라미터:
            config: 전체 애플리케이션 설정 객체
        """
        self._config = config

        # asyncio 큐 (크기는 config로 제한)
        self._audio_queue: asyncio.Queue[AudioPacket] = asyncio.Queue(
            maxsize=config.capture.audio_queue_size
        )
        self._video_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(
            maxsize=config.capture.video_queue_size
        )

        # 실행 상태
        self._running: bool = False

        # DeckLink 객체 (start() 시 초기화, stop() 시 해제)
        self._device: Optional[DeckLinkDevice]       = None
        self._input:  Optional[DeckLinkInput]        = None
        self._callback: Optional[DeckLinkInputCallback] = None

        # 프레임/패킷 카운터
        self._frame_id:  int = 0
        self._packet_id: int = 0

        # asyncio 이벤트 루프 참조 (콜백 스레드에서 사용)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            f"DeckLinkCapture 초기화: device_index={config.capture.device_index}, "
            f"video_mode={config.capture.video_mode}, "
            f"audio_channels={config.capture.audio_channels}"
        )

    # =========================================================================
    # 공개 인터페이스 (FileMockCapture와 동일)
    # =========================================================================

    def get_audio_queue(self) -> asyncio.Queue[AudioPacket]:
        """AudioPacket이 담기는 asyncio.Queue를 반환합니다."""
        return self._audio_queue

    def get_video_queue(self) -> asyncio.Queue[VideoFrame]:
        """VideoFrame이 담기는 asyncio.Queue를 반환합니다."""
        return self._video_queue

    async def start(self) -> None:
        """
        SDI 입력을 열고 비디오/오디오 스트리밍을 시작합니다.

        수행 순서:
            1. SDK 존재 확인
            2. 지정 장치 인덱스로 DeckLink 장치 열기
            3. IDeckLinkInput 인터페이스 획득
            4. 비디오/오디오 입력 활성화
            5. IDeckLinkInputCallback 등록
            6. 스트림 시작

        예외:
            DeckLinkSDKNotFoundError: SDK가 없을 때
            DeckLinkAPIError: 장치 열기/설정 실패 시
        """
        if self._running:
            logger.warning("DeckLinkCapture가 이미 실행 중입니다")
            return

        self._loop = asyncio.get_running_loop()

        # ── 1. 장치 열기 ────────────────────────────────────────────────────
        device_index = self._config.capture.device_index
        logger.info(f"DeckLink 장치 열기: index={device_index}")
        self._device = open_device(device_index)

        # ── 2. IDeckLinkInput 획득 ──────────────────────────────────────────
        self._input = self._device.GetInput()

        # ── 3. 비디오 입력 활성화 ───────────────────────────────────────────
        video_mode_str = self._config.capture.video_mode
        display_mode   = VIDEO_MODE_MAP.get(video_mode_str)
        if display_mode is None:
            supported = list(VIDEO_MODE_MAP.keys())
            raise DeckLinkAPIError(
                f"지원하지 않는 비디오 모드: '{video_mode_str}'. "
                f"지원 목록: {supported}"
            )

        pixel_fmt_str = self._config.capture.pixel_format
        pixel_format  = (
            BMDPixelFormat.YUV422_8bit
            if pixel_fmt_str == "yuv422"
            else BMDPixelFormat.BGRA_8bit
        )

        self._input.EnableVideoInput(display_mode, pixel_format)
        logger.info(
            f"비디오 입력 활성화: mode={video_mode_str}, "
            f"pixel_format={pixel_fmt_str} (0x{pixel_format:08X})"
        )

        # ── 4. 오디오 입력 활성화 ───────────────────────────────────────────
        bit_depth   = self._config.capture.audio_bit_depth
        channel_cnt = len(self._config.capture.audio_channels)
        sample_type = (
            BMDAudioSampleType.Int16bit
            if bit_depth == 16
            else BMDAudioSampleType.Int32bit
        )

        self._input.EnableAudioInput(
            BMDAudioSampleRate.Rate48kHz,
            sample_type,
            channel_cnt,
        )
        logger.info(
            f"오디오 입력 활성화: sample_rate=48000Hz, "
            f"bit_depth={bit_depth}bit, channels={channel_cnt}"
        )

        # ── 5. 콜백 등록 ────────────────────────────────────────────────────
        self._callback = DeckLinkInputCallback(
            on_frame_arrived=self._on_frame_arrived,
            audio_bit_depth=bit_depth,
            audio_channels=channel_cnt,
        )
        self._input.SetCallback(self._callback.get_ptr())
        logger.info("DeckLink 입력 콜백 등록 완료")

        # ── 6. 스트리밍 시작 ────────────────────────────────────────────────
        self._running = True
        self._input.StartStreams()
        logger.info("DeckLink SDI 캡처 시작")

    async def stop(self) -> None:
        """
        스트림을 중지하고 DeckLink 리소스를 해제합니다.

        수행 순서:
            1. StopStreams() 호출
            2. SetCallback(None) 으로 콜백 해제
            3. COM 객체 Release()
        """
        if not self._running:
            return

        self._running = False
        logger.info("DeckLink SDI 캡처 중지 시작")

        # 스트리밍 중지
        if self._input is not None:
            try:
                self._input.StopStreams()
            except Exception as exc:
                logger.warning(f"StopStreams 오류 (무시): {exc}")

            try:
                # 콜백 해제 (nullptr 전달)
                self._input.SetCallback(0)
            except Exception as exc:
                logger.warning(f"콜백 해제 오류 (무시): {exc}")

            self._input.Release()
            self._input = None

        if self._device is not None:
            self._device.Release()
            self._device = None

        # 콜백 객체 참조 해제 (GC 허용)
        self._callback = None
        self._loop     = None

        logger.info("DeckLink SDI 캡처 중지 완료")

    # =========================================================================
    # DeckLink SDK 콜백 (별도 스레드에서 호출됨)
    # =========================================================================

    def _on_frame_arrived(
        self,
        video: Optional[DeckLinkVideoFrame],
        audio: Optional[DeckLinkAudioPacket],
    ) -> None:
        """
        IDeckLinkInputCallback.VideoInputFrameArrived 구현체입니다.

        DeckLink SDK가 별도 스레드에서 호출합니다.
        asyncio-safe 방식으로 큐에 데이터를 전달하기 위해
        loop.call_soon_threadsafe()를 사용합니다.
        """
        if not self._running or self._loop is None:
            return

        timestamp_ns = time.time_ns()

        if video is not None:
            try:
                video_data = video.GetBytes()
                width      = video.GetWidth()
                height     = video.GetHeight()
                frame = VideoFrame(
                    frame_id=self._frame_id,
                    timestamp_ns=timestamp_ns,
                    width=width,
                    height=height,
                    pixel_format="yuv422",
                    data=video_data,
                )
                self._frame_id += 1
                self._loop.call_soon_threadsafe(
                    self._enqueue_video_sync, frame
                )
            except Exception as exc:
                logger.error(f"비디오 프레임 처리 오류: {exc}", exc_info=True)

        if audio is not None:
            try:
                bit_depth   = self._config.capture.audio_bit_depth
                channel_cnt = len(self._config.capture.audio_channels)
                audio_data  = audio.GetBytes(bit_depth, channel_cnt)
                sample_cnt  = audio.GetSampleCount()

                packet = AudioPacket(
                    packet_id=self._packet_id,
                    timestamp_ns=timestamp_ns,
                    sample_rate=BMDAudioSampleRate.Rate48kHz,
                    bit_depth=bit_depth,
                    channels=channel_cnt,
                    data=audio_data,
                )
                self._packet_id += 1
                self._loop.call_soon_threadsafe(
                    self._enqueue_audio_sync, packet
                )

                if self._packet_id % 100 == 0:
                    logger.debug(
                        f"오디오 패킷 {self._packet_id}개 처리 완료 "
                        f"(최근: {sample_cnt} samples)"
                    )
            except Exception as exc:
                logger.error(f"오디오 패킷 처리 오류: {exc}", exc_info=True)

    # =========================================================================
    # asyncio 루프 스레드에서 실행되는 동기 큐 삽입 함수
    # =========================================================================

    def _enqueue_video_sync(self, frame: VideoFrame) -> None:
        """asyncio 루프 스레드에서 video_queue에 프레임을 삽입합니다."""
        try:
            self._video_queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._video_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._video_queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning(f"비디오 큐 오버플로우: 프레임 {frame.frame_id} 드롭")

    def _enqueue_audio_sync(self, packet: AudioPacket) -> None:
        """asyncio 루프 스레드에서 audio_queue에 패킷을 삽입합니다."""
        try:
            self._audio_queue.put_nowait(packet)
        except asyncio.QueueFull:
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._audio_queue.put_nowait(packet)
            except asyncio.QueueFull:
                logger.warning(f"오디오 큐 오버플로우: 패킷 {packet.packet_id} 드롭")
