"""
파일 기반 모의 캡처 모듈입니다.

역할:
- WAV 파일을 실시간 속도로 스트리밍하여 AudioPacket 생성
- 검정 배경 VideoFrame을 30fps로 생성
- DeckLinkCapture와 동일한 asyncio.Queue 인터페이스 제공
- loop, playback_speed 설정으로 반복 재생 및 속도 제어 지원

사용 예시:
    >>> capture = FileMockCapture(config)
    >>> await capture.start()
    >>> audio_queue = capture.get_audio_queue()
    >>> packet = await audio_queue.get()
    >>> await capture.stop()
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.capture import AudioPacket, VideoFrame
from src.config.schema import AppConfig

# 모듈 로거
logger = logging.getLogger(__name__)

# 비디오 출력 고정 파라미터
_VIDEO_FPS = 30
_VIDEO_WIDTH = 1920
_VIDEO_HEIGHT = 1080
_VIDEO_PIXEL_FORMAT = "yuv422"


class FileMockCapture:
    """
    WAV 파일을 실시간 스트리밍으로 시뮬레이션하는 모의 캡처 클래스입니다.

    DeckLinkCapture와 동일한 인터페이스(get_audio_queue, get_video_queue)를 제공하여
    SDI 하드웨어 없이 전체 파이프라인 테스트가 가능합니다.

    오디오 생성 흐름:
        WAV 파일 로드 → chunk_size_ms 단위 분할 → AudioPacket 생성 → audio_queue에 put

    비디오 생성 흐름:
        검정 배경 프레임 생성 → 30fps 주기 → VideoFrame 생성 → video_queue에 put
    """

    def __init__(self, config: AppConfig) -> None:
        """
        FileMockCapture를 초기화합니다.

        파라미터:
            config (AppConfig): 전체 애플리케이션 설정 객체
        """
        self._config = config
        self._loop_enabled = config.capture.test_file.loop
        self._playback_speed = config.capture.test_file.playback_speed

        # video_path가 설정되면 ffmpeg로 오디오 추출, 없으면 audio_path(WAV) 직접 사용
        self._tmp_wav: Optional[tempfile.NamedTemporaryFile] = None
        video_path_str = config.capture.test_file.video_path
        if video_path_str:
            self._audio_path = self._extract_audio(Path(video_path_str))
        else:
            self._audio_path = Path(config.capture.test_file.audio_path)

        # asyncio 큐 (크기는 config로 제한)
        self._audio_queue: asyncio.Queue[AudioPacket] = asyncio.Queue(
            maxsize=config.capture.audio_queue_size
        )
        self._video_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(
            maxsize=config.capture.video_queue_size
        )

        # 실행 상태 플래그
        self._running: bool = False

        # 비동기 프로듀서 태스크
        self._audio_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None

        # 검정 배경 프레임 데이터 (한 번 생성 후 재사용)
        self._black_frame_data: bytes = _create_black_frame_bytes(
            _VIDEO_WIDTH, _VIDEO_HEIGHT
        )

        logger.info(
            f"FileMockCapture 초기화 완료: "
            f"audio={self._audio_path}, "
            f"loop={self._loop_enabled}, "
            f"playback_speed={self._playback_speed}x"
        )

    # =========================================================================
    # 공개 인터페이스
    # =========================================================================

    def get_audio_queue(self) -> asyncio.Queue[AudioPacket]:
        """AudioPacket이 담기는 asyncio.Queue를 반환합니다."""
        return self._audio_queue

    def get_video_queue(self) -> asyncio.Queue[VideoFrame]:
        """VideoFrame이 담기는 asyncio.Queue를 반환합니다."""
        return self._video_queue

    async def start(self) -> None:
        """
        오디오/비디오 프로듀서 태스크를 시작합니다.

        이미 실행 중이면 경고 로그를 출력하고 반환합니다.
        """
        if self._running:
            logger.warning("FileMockCapture가 이미 실행 중입니다")
            return

        self._running = True
        self._audio_task = asyncio.create_task(
            self._audio_producer(), name="mock_audio_producer"
        )
        self._video_task = asyncio.create_task(
            self._video_producer(), name="mock_video_producer"
        )

        logger.info("FileMockCapture 시작: 오디오/비디오 프로듀서 태스크 실행")

    async def stop(self) -> None:
        """
        프로듀서 태스크를 취소하고 리소스를 정리합니다.

        태스크 취소 후 완료를 대기하여 깔끔한 종료를 보장합니다.
        """
        if not self._running:
            return

        self._running = False
        logger.info("FileMockCapture 중지 시작")

        for task, name in [
            (self._audio_task, "오디오"),
            (self._video_task, "비디오"),
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug(f"{name} 프로듀서 태스크 종료")

        self._audio_task = None
        self._video_task = None

        # ffmpeg로 생성한 임시 WAV 파일 정리
        if self._tmp_wav is not None:
            try:
                self._tmp_wav.close()
            except Exception:
                pass
            self._tmp_wav = None

        logger.info("FileMockCapture 중지 완료")

    # =========================================================================
    # 내부 헬퍼 메서드
    # =========================================================================

    def _extract_audio(self, video_path: Path) -> Path:
        """
        ffmpeg로 비디오 파일에서 오디오를 추출하여 임시 WAV 파일로 저장합니다.

        파라미터:
            video_path: MP4/MOV/MKV 등 영상 파일 경로

        반환값:
            추출된 WAV 파일 경로
        """
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        self._tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(self._tmp_wav.name)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                   # 비디오 스트림 제외
            "-acodec", "pcm_s16le",  # 16bit PCM
            "-ar", "48000",          # 48kHz (SDI 원본 샘플레이트)
            "-ac", "2",              # 스테레오
            str(tmp_path),
        ]

        logger.info(f"ffmpeg 오디오 추출 시작: {video_path} → {tmp_path}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg 오디오 추출 실패 (returncode={result.returncode}): {err}"
            )

        logger.info(f"ffmpeg 오디오 추출 완료: {tmp_path}")
        return tmp_path

    # =========================================================================
    # 내부 프로듀서 메서드
    # =========================================================================

    async def _audio_producer(self) -> None:
        """
        WAV 파일을 읽어 AudioPacket을 생성하고 audio_queue에 추가하는 프로듀서입니다.

        chunk_size_ms 단위로 오디오를 분할하여 실시간 속도로 큐에 전달합니다.
        loop=True이면 파일 끝에 도달 시 처음부터 반복합니다.
        큐가 꽉 찬 경우 가장 오래된 패킷을 제거 후 새 패킷을 삽입합니다.
        """
        packet_id = 0
        chunk_size_ms = self._config.audio.chunk_size_ms

        while self._running:
            try:
                # WAV 파일 로드 (int16로 읽어서 bytes 변환 비용 최소화)
                audio_data, file_sample_rate = sf.read(
                    str(self._audio_path), dtype="int16", always_2d=False
                )

                # 채널 수 확인
                if audio_data.ndim == 1:
                    channels = 1
                else:
                    channels = audio_data.shape[1]

                # chunk_size_ms에 해당하는 샘플 수 계산
                samples_per_chunk = int(file_sample_rate * chunk_size_ms / 1000)
                total_samples = audio_data.shape[0]

                logger.info(
                    f"오디오 파일 로드: {self._audio_path}, "
                    f"sample_rate={file_sample_rate}Hz, "
                    f"channels={channels}, "
                    f"total_samples={total_samples}, "
                    f"chunk_size={samples_per_chunk}samples"
                )

                # 청크 단위 전송 시작
                stream_start_ns = time.time_ns()

                for chunk_index, chunk_start in enumerate(
                    range(0, total_samples, samples_per_chunk)
                ):
                    if not self._running:
                        return

                    chunk_end = min(chunk_start + samples_per_chunk, total_samples)

                    # 2D 배열이면 슬라이싱 방식 다름
                    if audio_data.ndim == 1:
                        chunk_data = audio_data[chunk_start:chunk_end]
                    else:
                        chunk_data = audio_data[chunk_start:chunk_end, :]

                    # timestamp: 스트림 시작 시각 + 청크 시작 샘플의 실제 시각
                    elapsed_samples_ns = int(
                        chunk_start / file_sample_rate * 1_000_000_000
                    )
                    timestamp_ns = stream_start_ns + elapsed_samples_ns

                    packet = AudioPacket(
                        packet_id=packet_id,
                        timestamp_ns=timestamp_ns,
                        sample_rate=file_sample_rate,
                        bit_depth=16,
                        channels=channels,
                        data=chunk_data.tobytes(),
                    )

                    await _put_with_overflow_drop(
                        self._audio_queue, packet, f"오디오 패킷 {packet_id}"
                    )

                    packet_id += 1

                    if packet_id % 100 == 0:
                        logger.debug(f"오디오 패킷 {packet_id}개 생성 완료")

                    # 실시간 속도 시뮬레이션
                    sleep_sec = (chunk_size_ms / 1000.0) / self._playback_speed
                    await asyncio.sleep(sleep_sec)

                # 파일 재생 완료
                logger.info(
                    f"오디오 파일 재생 완료: {self._audio_path}, "
                    f"총 {packet_id}개 패킷 생성"
                )

                if not self._loop_enabled:
                    self._running = False
                    return

                # loop=True: 처음부터 반복
                logger.debug("오디오 파일 반복 재생 시작")

            except FileNotFoundError:
                logger.error(f"오디오 파일을 찾을 수 없습니다: {self._audio_path}")
                self._running = False
                return

            except Exception as exc:
                logger.error(f"오디오 프로듀서 오류: {exc}", exc_info=True)
                if not self._loop_enabled:
                    self._running = False
                    return
                # loop 모드에서는 1초 대기 후 재시도
                await asyncio.sleep(1.0)

    async def _video_producer(self) -> None:
        """
        검정 배경 VideoFrame을 30fps 주기로 생성하고 video_queue에 추가하는 프로듀서입니다.

        미리 생성된 검정 프레임 데이터를 재사용하여 CPU 부하를 최소화합니다.
        큐가 꽉 찬 경우 가장 오래된 프레임을 제거 후 새 프레임을 삽입합니다.
        """
        frame_id = 0
        frame_interval_sec = 1.0 / _VIDEO_FPS / self._playback_speed

        logger.info(
            f"비디오 프로듀서 시작: {_VIDEO_FPS}fps, "
            f"interval={frame_interval_sec*1000:.1f}ms"
        )

        while self._running:
            timestamp_ns = time.time_ns()

            frame = VideoFrame(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                width=_VIDEO_WIDTH,
                height=_VIDEO_HEIGHT,
                pixel_format=_VIDEO_PIXEL_FORMAT,
                data=self._black_frame_data,
            )

            await _put_with_overflow_drop(
                self._video_queue, frame, f"비디오 프레임 {frame_id}"
            )

            frame_id += 1
            await asyncio.sleep(frame_interval_sec)


# =============================================================================
# 모듈 레벨 헬퍼 함수
# =============================================================================

def _create_black_frame_bytes(width: int, height: int) -> bytes:
    """
    YUV422(UYVY) 포맷의 검정 배경 프레임 데이터를 생성합니다.

    YUV422 UYVY 포맷: 2픽셀당 4바이트 [U, Y0, V, Y1]
    검정색 (ITU-R BT.601 limited range): Y=16, U=128, V=128

    파라미터:
        width: 프레임 가로 픽셀 수
        height: 프레임 세로 픽셀 수

    반환값:
        bytes: YUV422 포맷 프레임 데이터
    """
    total_macro_pixels = (width * height) // 2
    # UYVY: [U=128, Y0=16, V=128, Y1=16] 반복
    frame_array = np.empty(total_macro_pixels * 4, dtype=np.uint8)
    frame_array[0::4] = 128  # U
    frame_array[1::4] = 16   # Y0
    frame_array[2::4] = 128  # V
    frame_array[3::4] = 16   # Y1
    return frame_array.tobytes()


async def _put_with_overflow_drop(
    queue: asyncio.Queue,
    item: object,
    item_name: str,
) -> None:
    """
    큐가 꽉 찬 경우 가장 오래된 항목을 제거하고 새 항목을 삽입합니다.

    파라미터:
        queue: 대상 asyncio.Queue
        item: 삽입할 항목
        item_name: 로그 출력용 항목 이름
    """
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        # 가장 오래된 항목 제거
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        # 제거 후 재삽입 (여전히 가득 찬 경우에도 대기)
        await queue.put(item)
        logger.warning(f"큐 오버플로우: {item_name} 삽입을 위해 오래된 항목 제거")
