"""
SDI-RealtimeSubtitle 파이프라인 오케스트레이터

역할:
- 모든 모듈을 초기화하고 asyncio.gather로 병렬 실행
- 파이프라인: FileMockCapture → AudioResampler → STTStreamer → SubtitleManager → VideoCompositor
- SIGINT/SIGTERM 핸들러로 graceful shutdown
- 에러 발생 시 오류 로깅 후 종료

실행 예시:
    파일 모드 (STT 없이 자막 표시):
        python main.py --mode file --audio tests/fixtures/sample_audio.wav --no-stt

    파일 모드 (실제 STT 연동):
        python main.py --mode file --audio tests/fixtures/sample_audio.wav

    라이브 모드:
        python main.py --mode live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# 로깅 설정 (다른 임포트보다 먼저)
def _setup_logging(log_level: str, log_format: str, log_dir: str, session_id: str) -> None:
    """구조화 로깅을 설정합니다."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    handlers = []

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(funcName)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level"},
            )
        except ImportError:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # 파일 핸들러
    log_file = Path(log_dir) / f"session_{session_id}.jsonl"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)


class Pipeline:
    """
    전체 파이프라인을 관리하는 오케스트레이터 클래스입니다.

    파이프라인 구조:
        [FileMockCapture / DeckLinkCapture]
              │ audio_queue    │ video_queue
              ▼                ▼
        [AudioResampler]    [VideoCompositor] ◀──────────────┐
              │                                              │
              │ pcm_chunks                         SubtitleEvent
              ▼                                              │
        [ClovaSpeechStreamer]                    [SubtitleManager]
              │ result_queue                                 ▲
              └─────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config,
        no_stt: bool = False,
        no_display: bool = False,
        config_manager=None,
    ) -> None:
        from src.audio.resampler import AudioResampler
        from src.compositor.video_compositor import VideoCompositor
        from src.subtitle.subtitle_manager import SubtitleManager
        from src.metrics.metrics_store import MetricsStore
        from src.metrics.latency_tracker import LatencyTracker

        self._config = config
        self._no_stt = no_stt
        self._no_display = no_display

        # 핫스왑 설정 감시를 위한 ConfigManager (선택적)
        self._config_manager = config_manager

        # 메트릭 모듈 (대시보드와 지연시간 추적 공유)
        self._metrics_store = MetricsStore()
        self._latency_tracker = LatencyTracker(config)

        # 모듈 인스턴스
        self._capture = None
        self._resampler = AudioResampler(config)
        self._stt_streamer = None
        self._subtitle_manager = SubtitleManager(config)
        self._compositor = None

        # 파이프라인 상태
        self._status: str = "idle"  # "idle" | "running" | "stopping" | "error"

        # 종료 이벤트
        self._shutdown_event = asyncio.Event()

        # 실행 중 태스크 목록
        self._tasks: list[asyncio.Task] = []

    def get_status(self) -> str:
        """파이프라인의 현재 상태를 반환합니다."""
        return self._status

    async def run(self) -> None:
        """파이프라인을 시작하고 종료 신호를 기다립니다."""
        self._status = "running"
        self._shutdown_event = asyncio.Event()  # 재실행을 위해 이벤트 초기화
        logger.info("파이프라인 초기화 시작")

        # 캡처 모듈 초기화
        self._capture = self._create_capture()
        await self._capture.start()

        # 컴포지터 초기화
        from src.compositor.video_compositor import VideoCompositor
        self._compositor = VideoCompositor(
            self._config, self._capture.get_video_queue()
        )

        # STT 스트리머 초기화 (no_stt=False 시)
        if not self._no_stt:
            from src.stt.clova_streamer import ClovaSpeechStreamer
            self._stt_streamer = ClovaSpeechStreamer(self._config)
            try:
                await self._stt_streamer.connect()
                self._metrics_store.update_stt_status(connected=True)
            except Exception as exc:
                logger.error(f"STT 연결 실패: {exc}. --no-stt 옵션으로 재실행하세요.")
                self._metrics_store.update_stt_status(connected=False, error_count=1)
                await self._capture.stop()
                return

        logger.info("파이프라인 시작")

        # 모든 파이프라인 태스크 병렬 실행
        tasks = [
            asyncio.create_task(self._audio_pipeline(), name="audio_pipeline"),
            asyncio.create_task(self._video_pipeline(), name="video_pipeline"),
        ]

        if not self._no_stt:
            tasks.append(
                asyncio.create_task(self._stt_result_consumer(), name="stt_consumer")
            )

        self._tasks = tasks

        try:
            # 종료 신호 또는 태스크 완료 대기
            done, pending = await asyncio.wait(
                [asyncio.create_task(self._shutdown_event.wait())] + tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # 오류 확인
            for task in done:
                if task.exception():
                    logger.error(f"파이프라인 태스크 오류: {task.exception()}")

        finally:
            await self._shutdown()
            self._status = "idle"

    async def _shutdown(self) -> None:
        """파이프라인을 순서대로 종료합니다."""
        logger.info("파이프라인 종료 시작")

        # 실행 중인 태스크 취소
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # STT 연결 종료
        if self._stt_streamer:
            await self._stt_streamer.disconnect()

        # 캡처 종료
        if self._capture:
            await self._capture.stop()

        # 컴포지터 창 닫기
        if self._compositor:
            self._compositor.close()

        # 자막 파일 저장
        if self._subtitle_manager:
            history = self._subtitle_manager.flush_history()
            if history:
                output_dir = Path(self._config.subtitle.export.output_dir)
                self._subtitle_manager.export_srt(output_dir / "final_subtitles.srt")
                self._subtitle_manager.export_vtt(output_dir / "final_subtitles.vtt")
                logger.info(f"자막 파일 저장 완료: {output_dir}")

        # 핫스왑 감시 종료
        if self._config_manager is not None:
            self._config_manager.stop_watch()

        logger.info("파이프라인 종료 완료")

    def request_shutdown(self) -> None:
        """외부(시그널 핸들러 등)에서 종료를 요청합니다."""
        self._shutdown_event.set()

    def apply_config(self, old_config, new_config) -> None:
        """
        설정 변경을 파이프라인 모듈 전체에 즉시 적용합니다 (핫스왑).

        ConfigManager.subscribe()에 콜백으로 등록되어 파일 변경 시 자동 호출됩니다.

        적용 범위:
        - AudioResampler: VAD 설정(enabled, mode, padding_ms), gain_db
        - SubtitleManager: sync_offset_ms, display_duration_ms, show_partial
        - VideoCompositor: 폰트, 색상, 위치

        파라미터:
            old_config: 이전 설정 객체 (미사용, 서명 일치를 위해 포함)
            new_config: 새 설정 객체
        """
        self._config = new_config
        self._resampler.update_config(new_config)
        self._subtitle_manager.update_config(new_config)
        if self._compositor is not None:
            self._compositor.update_style(new_config)
        logger.info("파이프라인 설정 핫스왑 완료")

    # =========================================================================
    # 파이프라인 태스크
    # =========================================================================

    async def _audio_pipeline(self) -> None:
        """
        오디오 파이프라인: audio_queue → AudioResampler → STT 전송

        AudioPacket을 소비하여 PCMChunk로 변환 후 STT에 전송합니다.
        STT 비활성화 시 변환 결과를 버립니다.
        """
        audio_queue = self._capture.get_audio_queue()
        logger.info("오디오 파이프라인 시작")

        while not self._shutdown_event.is_set():
            try:
                packet = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # AudioPacket → PCMChunk 변환
            chunks = self._resampler.resample(packet)

            if chunks:
                # 오디오 레벨 MetricsStore 업데이트
                rms = max(c.rms for c in chunks)
                peak = max(c.peak for c in chunks)
                self._metrics_store.update_audio_level(rms, peak)

                # 캡처 타임스탬프 기록 (청크 단위)
                for chunk in chunks:
                    self._latency_tracker.record_capture(
                        chunk.chunk_id, chunk.capture_timestamp_ns
                    )

            # STT 전송
            if self._stt_streamer and chunks:
                for chunk in chunks:
                    self._latency_tracker.record_stt_send(
                        chunk.chunk_id, time.time_ns()
                    )
                    await self._stt_streamer.stream_audio(chunk)

        logger.info("오디오 파이프라인 종료")

    async def _video_pipeline(self) -> None:
        """
        비디오 파이프라인: video_queue → VideoCompositor → 화면 출력

        VideoFrame을 소비하여 현재 자막을 오버레이하고 화면에 출력합니다.
        """
        video_queue = self._capture.get_video_queue()
        logger.info("비디오 파이프라인 시작")

        while not self._shutdown_event.is_set():
            try:
                frame = await asyncio.wait_for(video_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # 현재 자막 조회
            subtitle_event = self._subtitle_manager.get_current_subtitle()

            # 자막 오버레이
            composited = self._compositor.composite(frame, subtitle_event)

            # 프레임 통계 MetricsStore 업데이트
            buf_status = self._compositor.get_buffer_status()
            self._metrics_store.update_frame_stats(
                total_frames=self._compositor._total_frames,
                drop_count=buf_status.drop_count,
                drop_rate=buf_status.drop_rate,
                queue_depth=buf_status.queue_depth,
            )

            # 화면 출력 (no_display 옵션 시 건너뜀)
            if not self._no_display:
                should_continue = self._compositor.display(composited)
                if not should_continue:
                    logger.info("사용자가 프리뷰 창을 닫아 파이프라인 종료")
                    self.request_shutdown()
                    return

        logger.info("비디오 파이프라인 종료")

    async def _stt_result_consumer(self) -> None:
        """
        STT 결과 소비자: result_queue → SubtitleManager

        STTResult를 소비하여 SubtitleManager에 전달합니다.
        None(종료 신호)을 받으면 종료합니다.
        """
        result_queue = self._stt_streamer.get_result_queue()
        logger.info("STT 결과 소비자 시작")

        while not self._shutdown_event.is_set():
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if result is None:
                logger.info("STT 종료 신호 수신")
                self.request_shutdown()
                return

            # STT 수신 타임스탬프 기록
            self._latency_tracker.record_stt_receive(
                result_id=result.result_id,
                packet_id=result.last_chunk_id,
                timestamp_ns=result.receive_timestamp_ns,
                result_type=result.type,
            )

            # SubtitleManager에 전달
            self._subtitle_manager.process_result(result)

            # 현재 자막 MetricsStore 업데이트
            self._metrics_store.update_subtitle(result.text, result.type == "partial")

            # STT 연결 상태 업데이트
            self._metrics_store.update_stt_status(connected=True)

        logger.info("STT 결과 소비자 종료")

    # =========================================================================
    # 내부 헬퍼
    # =========================================================================

    def _create_capture(self):
        """설정에 따라 적절한 캡처 모듈을 생성합니다."""
        if self._config.system.mode == "file":
            from src.capture.file_mock_capture import FileMockCapture
            return FileMockCapture(self._config)
        else:
            from src.capture.decklink_capture import DeckLinkCapture
            return DeckLinkCapture(self._config)


# =============================================================================
# 진입점
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="SDI-RealtimeSubtitle: 실시간 자막 생성 시스템"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="설정 파일 경로 (기본: config.yaml)"
    )
    parser.add_argument(
        "--mode", choices=["live", "file"], help="실행 모드 (config.yaml 오버라이드)"
    )
    parser.add_argument(
        "--audio", help="테스트 오디오 파일 경로 (file 모드 전용)"
    )
    parser.add_argument(
        "--no-stt", action="store_true", help="STT 없이 파이프라인 실행 (개발 테스트용)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="OpenCV 화면 출력 비활성화"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="실행 시간 제한 (초, 0=무제한)"
    )
    parser.add_argument(
        "--web-dashboard", action="store_true", help="웹 대시보드 활성화 (기본 포트: 8765)"
    )
    parser.add_argument(
        "--web-port", type=int, default=8765, help="웹 대시보드 포트 (기본: 8765)"
    )
    return parser.parse_args()


async def _run_with_timeout(pipeline: Pipeline, duration_sec: int) -> None:
    """파이프라인을 duration_sec 초 후에 자동 종료합니다."""
    if duration_sec > 0:
        await asyncio.sleep(duration_sec)
        logger.info(f"{duration_sec}초 경과, 파이프라인 자동 종료")
        pipeline.request_shutdown()


async def _main() -> None:
    """비동기 메인 함수입니다."""
    args = _parse_args()

    # 설정 로드
    from src.config.config_manager import ConfigManager
    manager = ConfigManager()
    config = manager.load(args.config)

    # 커맨드라인 오버라이드
    if args.mode:
        # 설정 dict를 재구성 (Pydantic 모델은 불변이므로 재생성)
        config_dict = config.model_dump()
        config_dict["system"]["mode"] = args.mode
        if args.audio:
            config_dict["capture"]["test_file"]["audio_path"] = args.audio
        from src.config.schema import AppConfig
        config = AppConfig(**config_dict)
    elif args.audio:
        config_dict = config.model_dump()
        config_dict["capture"]["test_file"]["audio_path"] = args.audio
        from src.config.schema import AppConfig
        config = AppConfig(**config_dict)

    # 세션 ID 생성
    session_id = config.system.session_id or uuid.uuid4().hex[:8]

    # 로깅 설정
    _setup_logging(
        config.system.log_level,
        config.system.log_format,
        config.system.log_dir,
        session_id,
    )

    logger.info(
        f"SDI-RealtimeSubtitle 시작: "
        f"session_id={session_id}, "
        f"mode={config.system.mode}"
    )

    # 파이프라인 생성
    pipeline = Pipeline(
        config,
        no_stt=args.no_stt,
        no_display=args.no_display,
        config_manager=manager,
    )

    # 핫스왑 설정 감시 등록
    manager.subscribe(pipeline.apply_config)
    manager.watch()

    # SIGINT/SIGTERM 핸들러 등록 (asyncio-safe 방식)
    loop = asyncio.get_event_loop()

    # 웹 대시보드 모드에서 메인 루프를 제어하는 종료 이벤트
    _main_shutdown = asyncio.Event()

    def _signal_handler():
        logger.info("종료 시그널 수신")
        pipeline.request_shutdown()
        _main_shutdown.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    web_dashboard = None
    if args.web_dashboard:
        from src.dashboard.web_dashboard import WebDashboard
        web_dashboard = WebDashboard(
            metrics_store=pipeline._metrics_store,
            pipeline=pipeline,
            config=config,
            host=config.dashboard.web.host,
            port=args.web_port,
        )

    # 웹 대시보드 시작
    if web_dashboard:
        await web_dashboard.start()

    # 파이프라인 실행
    try:
        if args.web_dashboard:
            # 웹 대시보드 모드: 파이프라인을 UI에서 시작/중지
            await _main_shutdown.wait()
        elif args.duration > 0:
            await asyncio.gather(
                pipeline.run(),
                _run_with_timeout(pipeline, args.duration),
            )
        else:
            await pipeline.run()
    finally:
        if web_dashboard:
            await web_dashboard.stop()

    logger.info("SDI-RealtimeSubtitle 종료")


if __name__ == "__main__":
    asyncio.run(_main())
