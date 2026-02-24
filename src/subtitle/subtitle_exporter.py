"""
자막 파일 내보내기 모듈입니다.

역할:
- SubtitleEvent 히스토리를 SRT 포맷 파일로 저장
- SubtitleEvent 히스토리를 WebVTT 포맷 파일로 저장

사용 예시:
    >>> exporter = SubtitleExporter()
    >>> exporter.export_srt(history, "output/subtitles/session.srt")
    >>> exporter.export_vtt(history, "output/subtitles/session.vtt")
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.subtitle import SubtitleEvent

logger = logging.getLogger(__name__)


class SubtitleExporter:
    """
    자막 이벤트를 SRT/VTT 파일로 내보내는 클래스입니다.

    파일 저장 실패 시 IOError/OSError를 상위로 전파합니다.
    """

    def export_srt(self, history: list[SubtitleEvent], filepath: str | Path) -> None:
        """
        SubtitleEvent 히스토리를 SRT 포맷으로 저장합니다.

        SRT 포맷:
            번호
            시작시간 --> 종료시간
            자막텍스트
            (빈 줄)

        파라미터:
            history: 저장할 SubtitleEvent 목록 (final 결과만 권장)
            filepath: 저장할 .srt 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # final 결과만 필터링 (partial 제외)
        final_events = [e for e in history if not e.is_partial]

        # 세션 시작 기준 상대 타임코드 계산 (첫 자막의 display_at_ns 기준)
        session_start_ns = final_events[0].display_at_ns if final_events else 0

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for index, event in enumerate(final_events, start=1):
                    start_str = _ns_to_srt_time(event.display_at_ns, session_start_ns)
                    end_str = _ns_to_srt_time(event.expire_at_ns, session_start_ns)

                    f.write(f"{index}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{event.text}\n")
                    f.write("\n")

            logger.info(f"SRT 파일 저장 완료: {filepath} ({len(final_events)}개 자막)")

        except OSError as exc:
            logger.error(f"SRT 파일 저장 실패: {filepath}, 오류: {exc}")
            raise

    def export_vtt(self, history: list[SubtitleEvent], filepath: str | Path) -> None:
        """
        SubtitleEvent 히스토리를 WebVTT 포맷으로 저장합니다.

        WebVTT 포맷:
            WEBVTT
            (빈 줄)
            시작시간 --> 종료시간
            자막텍스트
            (빈 줄)

        파라미터:
            history: 저장할 SubtitleEvent 목록 (final 결과만 권장)
            filepath: 저장할 .vtt 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        final_events = [e for e in history if not e.is_partial]

        # 세션 시작 기준 상대 타임코드 계산 (첫 자막의 display_at_ns 기준)
        session_start_ns = final_events[0].display_at_ns if final_events else 0

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")

                for event in final_events:
                    start_str = _ns_to_vtt_time(event.display_at_ns, session_start_ns)
                    end_str = _ns_to_vtt_time(event.expire_at_ns, session_start_ns)

                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{event.text}\n")
                    f.write("\n")

            logger.info(f"VTT 파일 저장 완료: {filepath} ({len(final_events)}개 자막)")

        except OSError as exc:
            logger.error(f"VTT 파일 저장 실패: {filepath}, 오류: {exc}")
            raise

    def append_srt(
        self,
        events: list[SubtitleEvent],
        filepath: str | Path,
        start_index: int,
        session_start_ns: int,
        *,
        truncate: bool = False,
    ) -> None:
        """
        새 이벤트를 SRT 파일에 추가합니다 (증분 쓰기).

        파라미터:
            events: 추가할 SubtitleEvent 목록
            filepath: 저장할 .srt 파일 경로
            start_index: 첫 이벤트의 1-based SRT 번호
            session_start_ns: 세션 시작 기준 nanoseconds (상대 타임코드 계산용)
            truncate: True이면 파일을 새로 생성 (기존 내용 삭제), False이면 추가
        """
        if not events:
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate else "a"

        try:
            with open(filepath, mode, encoding="utf-8") as f:
                for i, event in enumerate(events):
                    index = start_index + i
                    start_str = _ns_to_srt_time(event.display_at_ns, session_start_ns)
                    end_str = _ns_to_srt_time(event.expire_at_ns, session_start_ns)

                    f.write(f"{index}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{event.text}\n")
                    f.write("\n")

            logger.info(f"SRT 파일 추가 완료: {filepath} (+{len(events)}개)")

        except OSError as exc:
            logger.error(f"SRT 파일 추가 실패: {filepath}, 오류: {exc}")
            raise

    def append_vtt(
        self,
        events: list[SubtitleEvent],
        filepath: str | Path,
        session_start_ns: int,
        *,
        truncate: bool = False,
    ) -> None:
        """
        새 이벤트를 VTT 파일에 추가합니다 (증분 쓰기).

        파라미터:
            events: 추가할 SubtitleEvent 목록
            filepath: 저장할 .vtt 파일 경로
            session_start_ns: 세션 시작 기준 nanoseconds (상대 타임코드 계산용)
            truncate: True이면 파일을 새로 생성 (WEBVTT 헤더 포함), False이면 추가
        """
        if not events and not truncate:
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate else "a"

        try:
            with open(filepath, mode, encoding="utf-8") as f:
                if truncate:
                    f.write("WEBVTT\n\n")

                for event in events:
                    start_str = _ns_to_vtt_time(event.display_at_ns, session_start_ns)
                    end_str = _ns_to_vtt_time(event.expire_at_ns, session_start_ns)

                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{event.text}\n")
                    f.write("\n")

            logger.info(f"VTT 파일 추가 완료: {filepath} (+{len(events)}개)")

        except OSError as exc:
            logger.error(f"VTT 파일 추가 실패: {filepath}, 오류: {exc}")
            raise


# =============================================================================
# 헬퍼 함수
# =============================================================================

def _ns_to_srt_time(timestamp_ns: int, session_start_ns: int = 0) -> str:
    """
    nanoseconds 타임스탬프를 SRT 시간 포맷으로 변환합니다.

    SRT 포맷: HH:MM:SS,mmm

    파라미터:
        timestamp_ns: nanoseconds 타임스탬프
        session_start_ns: 세션 시작 기준 nanoseconds (상대 타임코드 계산용)

    반환값:
        str: SRT 시간 문자열
    """
    total_ms = max(0, (timestamp_ns - session_start_ns)) // 1_000_000
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minute = total_min % 60
    hour = total_min // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def _ns_to_vtt_time(timestamp_ns: int, session_start_ns: int = 0) -> str:
    """
    nanoseconds 타임스탬프를 WebVTT 시간 포맷으로 변환합니다.

    VTT 포맷: HH:MM:SS.mmm

    파라미터:
        timestamp_ns: nanoseconds 타임스탬프
        session_start_ns: 세션 시작 기준 nanoseconds (상대 타임코드 계산용)

    반환값:
        str: VTT 시간 문자열
    """
    # SRT와 동일하지만 구분자가 콤마(,) 대신 점(.)
    return _ns_to_srt_time(timestamp_ns, session_start_ns).replace(",", ".")
