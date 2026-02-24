"""
자막 매니저 모듈입니다.

역할:
- STT 결과(partial/final)를 수신하여 현재 표시할 자막 텍스트 결정
- sync_offset_ms를 적용하여 display_at_ns/expire_at_ns 계산
- partial 결과는 즉시 덮어쓰고, final 결과는 히스토리에 영구 저장
- VideoCompositor에 전달할 SubtitleEvent 반환
- SRT/VTT 파일 저장 지원
- 핫스왑 설정 업데이트 지원

사용 예시:
    >>> manager = SubtitleManager(config)
    >>> manager.process_result(stt_result)
    >>> event = manager.get_current_subtitle()
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from src.config.schema import AppConfig, SubtitleConfig
from src.stt import STTResult
from src.subtitle import SubtitleEvent
from src.subtitle.subtitle_exporter import SubtitleExporter

logger = logging.getLogger(__name__)


class SubtitleManager:
    """
    STT 결과를 자막 이벤트로 변환하고 관리하는 매니저 클래스입니다.

    partial/final 처리 전략:
    - partial: 현재 partial 버퍼를 덮어씀. expire_at은 현재 시각 + display_duration
    - final: partial 버퍼를 클리어하고 히스토리에 추가. expire_at 고정

    스레드 안전성:
    - get_current_subtitle()은 다른 스레드에서 호출될 수 있으므로 _lock으로 보호
    """

    def __init__(self, config: AppConfig) -> None:
        """
        SubtitleManager를 초기화합니다.

        파라미터:
            config (AppConfig): 전체 애플리케이션 설정 객체
        """
        self._subtitle_cfg = config.subtitle
        self._lock = threading.RLock()

        # 현재 표시 중인 partial 이벤트 (final이 오면 교체)
        self._current_partial: Optional[SubtitleEvent] = None

        # 확정된 final 자막 히스토리 버퍼
        self._history: list[SubtitleEvent] = []

        # 이벤트 순번
        self._event_id: int = 0

        # 파일 내보내기
        self._exporter = SubtitleExporter()

        # 증분 파일 쓰기 추적
        self._total_finals: int = 0
        self._written_counts: dict[str, int] = {}
        self._session_start_ns: Optional[int] = None

        logger.info(
            f"SubtitleManager 초기화: "
            f"sync_offset={self._subtitle_cfg.sync_offset_ms}ms, "
            f"display_duration={self._subtitle_cfg.display_duration_ms}ms, "
            f"show_partial={self._subtitle_cfg.show_partial}"
        )

    # =========================================================================
    # 공개 인터페이스
    # =========================================================================

    def process_result(self, stt_result: STTResult) -> SubtitleEvent:
        """
        STT 결과를 처리하여 SubtitleEvent를 생성하고 내부 상태를 업데이트합니다.

        partial 처리:
            - 현재 partial 버퍼를 새 텍스트로 덮어씀
            - expire_at = 현재 시각 + display_duration (상시 갱신)

        final 처리:
            - partial 버퍼 초기화
            - 히스토리 버퍼에 추가
            - 히스토리가 history_size를 초과하면 가장 오래된 항목 제거

        파라미터:
            stt_result (STTResult): 처리할 STT 결과

        반환값:
            SubtitleEvent: 생성된 자막 이벤트
        """
        now_ns = time.time_ns()
        sync_offset_ns = self._subtitle_cfg.sync_offset_ms * 1_000_000
        display_duration_ns = self._subtitle_cfg.display_duration_ms * 1_000_000

        display_at_ns = now_ns + sync_offset_ns
        expire_at_ns = display_at_ns + display_duration_ns
        is_partial = stt_result.type == "partial"

        event = SubtitleEvent(
            event_id=self._event_id,
            text=stt_result.text,
            is_partial=is_partial,
            display_at_ns=display_at_ns,
            expire_at_ns=expire_at_ns,
            font_path=self._subtitle_cfg.font.path,
            font_size=self._subtitle_cfg.font.size,
            color=self._subtitle_cfg.font.color,
            background_color=self._subtitle_cfg.font.background_color,
            stroke_color=self._subtitle_cfg.font.stroke_color,
            stroke_width=self._subtitle_cfg.font.stroke_width,
            position_x=self._subtitle_cfg.position.x,
            position_y=self._subtitle_cfg.position.y,
            anchor=self._subtitle_cfg.position.anchor,
        )
        self._event_id += 1

        should_export = False
        with self._lock:
            if is_partial:
                self._current_partial = event
                logger.debug(f"partial 자막 갱신: '{event.text[:30]}...'")

            else:
                # final: partial 클리어 + 히스토리 추가
                self._current_partial = None
                self._history.append(event)
                self._total_finals += 1

                if self._session_start_ns is None:
                    self._session_start_ns = event.display_at_ns

                # 히스토리 크기 제한
                max_history = self._subtitle_cfg.history_size
                if len(self._history) > max_history:
                    removed = self._history.pop(0)
                    logger.debug(f"히스토리 오버플로우: event_id={removed.event_id} 제거")

                logger.info(
                    f"final 자막 확정: '{event.text[:50]}', "
                    f"history_size={len(self._history)}"
                )

                should_export = self._subtitle_cfg.export.enabled

        # 파일 I/O는 lock 바깥에서 수행 (lock 점유 시간 최소화)
        if should_export:
            self._auto_export()

        return event

    def get_current_subtitle(self) -> Optional[SubtitleEvent]:
        """
        현재 시각 기준으로 표시해야 할 자막 이벤트를 반환합니다.

        판단 기준:
        1. partial 자막이 있으면 partial 반환 (show_partial=True 시)
        2. 없으면 히스토리의 마지막 final 자막 중 아직 만료되지 않은 것 반환
        3. 만료된 자막은 None 반환

        반환값:
            Optional[SubtitleEvent]: 표시할 자막 (없으면 None)
        """
        now_ns = time.time_ns()

        with self._lock:
            # 1. partial 자막 확인
            if (
                self._subtitle_cfg.show_partial
                and self._current_partial is not None
                and now_ns < self._current_partial.expire_at_ns
            ):
                return self._current_partial

            # 2. 마지막 final 자막 확인 (만료 여부 체크)
            if self._history:
                last_final = self._history[-1]
                if now_ns < last_final.expire_at_ns:
                    return last_final

        return None

    def flush_history(self, n: Optional[int] = None) -> list[SubtitleEvent]:
        """
        자막 히스토리 버퍼를 조회합니다.

        파라미터:
            n: 최근 n개를 반환 (None이면 전체)

        반환값:
            list[SubtitleEvent]: 히스토리 목록 (복사본)
        """
        with self._lock:
            if n is None:
                return list(self._history)
            return list(self._history[-n:])

    def export_srt(self, filepath: str | Path) -> None:
        """
        현재 히스토리를 SRT 파일로 저장합니다.

        파라미터:
            filepath: 저장할 .srt 파일 경로
        """
        with self._lock:
            history_copy = list(self._history)

        self._exporter.export_srt(history_copy, filepath)

    def export_vtt(self, filepath: str | Path) -> None:
        """
        현재 히스토리를 VTT 파일로 저장합니다.

        파라미터:
            filepath: 저장할 .vtt 파일 경로
        """
        with self._lock:
            history_copy = list(self._history)

        self._exporter.export_vtt(history_copy, filepath)

    def update_config(self, config: AppConfig) -> None:
        """
        설정을 핫스왑으로 업데이트합니다. 서비스 중단 없이 즉시 적용됩니다.

        업데이트 가능 항목:
        - 폰트 (경로, 크기, 색상)
        - 위치 (x, y, anchor)
        - sync_offset_ms, display_duration_ms
        - show_partial

        파라미터:
            config (AppConfig): 새 설정 객체
        """
        self._subtitle_cfg = config.subtitle
        logger.info(
            f"SubtitleManager 설정 핫스왑: "
            f"font_size={config.subtitle.font.size}, "
            f"sync_offset={config.subtitle.sync_offset_ms}ms"
        )

    # =========================================================================
    # 내부 메서드
    # =========================================================================

    def _auto_export(self) -> None:
        """
        설정에 지정된 포맷으로 자막 파일을 자동 저장합니다 (증분 쓰기).

        final 이벤트 발생 시마다 새 항목만 append 모드로 기록하여
        히스토리 누적에 따른 디스크 I/O 선형 증가를 방지합니다.
        lock을 짧게 잡아 히스토리 스냅샷만 취한 뒤 파일 I/O를 수행합니다.
        저장 실패 시 오류를 로깅하고 계속 진행합니다.
        """
        output_dir = Path(self._subtitle_cfg.export.output_dir)
        with self._lock:
            history_copy = list(self._history)
            total_finals = self._total_finals
            session_start_ns = self._session_start_ns

        if not history_copy or session_start_ns is None:
            return

        # 히스토리에서 제거된 이벤트 수 = 전체 final 수 - 현재 히스토리 크기
        removed_count = total_finals - len(history_copy)

        for fmt in self._subtitle_cfg.export.format:
            try:
                filepath = output_dir / f"subtitles.{fmt}"
                already_written = self._written_counts.get(fmt, 0)

                # 아직 파일에 기록되지 않은 이벤트의 히스토리 내 시작 인덱스
                new_start_in_history = already_written - removed_count
                new_events = history_copy[new_start_in_history:]

                if not new_events:
                    continue

                is_first_write = (already_written == 0)

                if fmt == "srt":
                    self._exporter.append_srt(
                        new_events, filepath,
                        start_index=already_written + 1,
                        session_start_ns=session_start_ns,
                        truncate=is_first_write,
                    )
                elif fmt == "vtt":
                    self._exporter.append_vtt(
                        new_events, filepath,
                        session_start_ns=session_start_ns,
                        truncate=is_first_write,
                    )
                else:
                    logger.warning(f"지원하지 않는 자막 포맷: {fmt}")
                    continue

                self._written_counts[fmt] = already_written + len(new_events)

            except Exception as exc:
                logger.error(f"자막 자동 저장 실패 ({fmt}): {exc}")
