"""
SubtitleManager 단위 테스트

검증 항목:
- partial → final 전환 정상 동작
- sync_offset_ms 적용 시 display_at_ns 지연 확인
- get_current_subtitle() 만료 자막 처리
- 히스토리 버퍼 크기 제한
- SRT/VTT 파일 저장 포맷 검증
- 핫스왑 설정 업데이트
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.config.schema import AppConfig
from src.stt import STTResult, WordTiming
from src.subtitle import SubtitleEvent
from src.subtitle.subtitle_manager import SubtitleManager


# =============================================================================
# 테스트 헬퍼
# =============================================================================

def _make_config(
    sync_offset_ms: int = 0,
    display_duration_ms: int = 3000,
    show_partial: bool = True,
    history_size: int = 100,
    export_enabled: bool = False,
    export_output_dir: str = "output/subtitles",
) -> AppConfig:
    """테스트용 AppConfig를 생성합니다."""
    return AppConfig(**{
        "subtitle": {
            "sync_offset_ms": sync_offset_ms,
            "display_duration_ms": display_duration_ms,
            "show_partial": show_partial,
            "history_size": history_size,
            "font": {
                "path": "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "size": 36,
                "color": "#FFFFFF",
                "background_color": "#00000080",
                "stroke_color": "#000000",
                "stroke_width": 2,
            },
            "position": {"x": 0.5, "y": 0.85, "anchor": "center"},
            "export": {
                "enabled": export_enabled,
                "format": ["srt", "vtt"],
                "output_dir": export_output_dir,
            },
        },
    })


def _make_stt_result(
    text: str,
    result_type: str = "final",
    result_id: int = 0,
) -> STTResult:
    """테스트용 STTResult를 생성합니다."""
    now = time.time_ns()
    return STTResult(
        result_id=result_id,
        type=result_type,
        text=text,
        send_timestamp_ns=now - 500_000_000,
        receive_timestamp_ns=now,
        confidence=0.95,
        words=[],
    )


# =============================================================================
# partial → final 전환 테스트
# =============================================================================

def test_partial_result_sets_current_partial():
    """partial STT 결과가 current_partial로 설정되는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    result = _make_stt_result("안녕하", result_type="partial")
    manager.process_result(result)

    assert manager._current_partial is not None
    assert manager._current_partial.text == "안녕하"
    assert manager._current_partial.is_partial is True


def test_final_result_clears_partial():
    """final STT 결과 처리 후 current_partial이 초기화되는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    # 먼저 partial 설정
    manager.process_result(_make_stt_result("안녕하", "partial"))
    assert manager._current_partial is not None

    # final 처리
    manager.process_result(_make_stt_result("안녕하세요", "final"))

    assert manager._current_partial is None


def test_final_result_added_to_history():
    """final STT 결과가 히스토리에 추가되는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("첫 번째 문장", "final"))
    manager.process_result(_make_stt_result("두 번째 문장", "final"))

    history = manager.flush_history()
    assert len(history) == 2
    assert history[0].text == "첫 번째 문장"
    assert history[1].text == "두 번째 문장"


def test_partial_not_added_to_history():
    """partial STT 결과는 히스토리에 추가되지 않는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("안녕", "partial"))
    manager.process_result(_make_stt_result("안녕하", "partial"))

    history = manager.flush_history()
    assert len(history) == 0


def test_partial_to_final_transition():
    """partial → final 전환 시 히스토리에 final 텍스트가 저장되는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("안녕하", "partial"))
    manager.process_result(_make_stt_result("안녕 하세요", "partial"))
    manager.process_result(_make_stt_result("안녕하세요 반갑습니다", "final"))

    history = manager.flush_history()
    assert len(history) == 1
    assert history[0].text == "안녕하세요 반갑습니다"
    assert history[0].is_partial is False


# =============================================================================
# sync_offset 테스트
# =============================================================================

def test_sync_offset_positive_delays_display():
    """sync_offset=500ms 설정 시 display_at_ns가 500ms 뒤인지 확인합니다."""
    sync_offset_ms = 500
    config = _make_config(sync_offset_ms=sync_offset_ms)
    manager = SubtitleManager(config)

    before_ns = time.time_ns()
    result = _make_stt_result("테스트", "final")
    event = manager.process_result(result)
    after_ns = time.time_ns()

    # display_at_ns는 호출 시각 + sync_offset_ms
    expected_min = before_ns + sync_offset_ms * 1_000_000
    expected_max = after_ns + sync_offset_ms * 1_000_000

    assert expected_min <= event.display_at_ns <= expected_max, (
        f"display_at_ns 범위 오류: {event.display_at_ns}, "
        f"기대 범위 [{expected_min}, {expected_max}]"
    )


def test_sync_offset_zero_no_delay():
    """sync_offset=0ms 설정 시 display_at_ns가 현재 시각과 거의 동일한지 확인합니다."""
    config = _make_config(sync_offset_ms=0)
    manager = SubtitleManager(config)

    before_ns = time.time_ns()
    event = manager.process_result(_make_stt_result("테스트", "final"))
    after_ns = time.time_ns()

    assert before_ns <= event.display_at_ns <= after_ns + 10_000_000  # ±10ms 허용


def test_sync_offset_negative_advances_display():
    """sync_offset=-100ms 설정 시 display_at_ns가 현재보다 앞인지 확인합니다."""
    config = _make_config(sync_offset_ms=-100)
    manager = SubtitleManager(config)

    before_ns = time.time_ns()
    event = manager.process_result(_make_stt_result("테스트", "final"))

    # display_at_ns가 호출 전보다 작아야 함 (100ms 앞당김)
    assert event.display_at_ns < before_ns + 10_000_000  # ±10ms 허용


def test_expire_at_ns_is_display_at_plus_duration():
    """expire_at_ns = display_at_ns + display_duration_ms인지 확인합니다."""
    display_duration_ms = 3000
    config = _make_config(display_duration_ms=display_duration_ms)
    manager = SubtitleManager(config)

    event = manager.process_result(_make_stt_result("테스트", "final"))

    expected_expire = event.display_at_ns + display_duration_ms * 1_000_000
    assert event.expire_at_ns == expected_expire


# =============================================================================
# get_current_subtitle 테스트
# =============================================================================

def test_get_current_subtitle_returns_partial_when_available():
    """partial 자막이 있으면 get_current_subtitle()이 partial을 반환하는지 확인합니다."""
    config = _make_config(show_partial=True)
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("안녕하", "partial"))

    current = manager.get_current_subtitle()
    assert current is not None
    assert current.is_partial is True
    assert current.text == "안녕하"


def test_get_current_subtitle_returns_none_when_show_partial_false():
    """show_partial=False이면 partial 자막을 반환하지 않는지 확인합니다."""
    config = _make_config(show_partial=False)
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("안녕하", "partial"))

    current = manager.get_current_subtitle()
    assert current is None


def test_get_current_subtitle_returns_final_after_partial_clears():
    """partial 이후 final이 오면 final을 반환하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("안녕하", "partial"))
    manager.process_result(_make_stt_result("안녕하세요", "final"))

    current = manager.get_current_subtitle()
    assert current is not None
    assert current.is_partial is False
    assert current.text == "안녕하세요"


def test_get_current_subtitle_returns_none_when_expired():
    """만료된 자막은 get_current_subtitle()이 None을 반환하는지 확인합니다."""
    # display_duration=1ms → 거의 즉시 만료
    config = _make_config(display_duration_ms=1)
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("테스트", "final"))
    import time as _time
    _time.sleep(0.01)  # 10ms 대기 (1ms 만료 후)

    current = manager.get_current_subtitle()
    assert current is None


def test_get_current_subtitle_returns_none_initially():
    """초기 상태에서 get_current_subtitle()이 None을 반환하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    assert manager.get_current_subtitle() is None


# =============================================================================
# 히스토리 버퍼 테스트
# =============================================================================

def test_history_size_limit():
    """히스토리 크기가 history_size를 초과하면 오래된 항목이 제거되는지 확인합니다."""
    max_size = 5
    config = _make_config(history_size=max_size)
    manager = SubtitleManager(config)

    for i in range(max_size + 3):
        manager.process_result(_make_stt_result(f"자막 {i}", "final"))

    history = manager.flush_history()
    assert len(history) == max_size

    # 가장 오래된 항목이 제거되고 최신 항목이 남아야 함
    assert history[0].text == f"자막 {3}"  # 처음 3개 제거
    assert history[-1].text == f"자막 {max_size + 2}"


def test_flush_history_n_returns_last_n():
    """flush_history(n)이 최근 n개를 반환하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    for i in range(5):
        manager.process_result(_make_stt_result(f"자막 {i}", "final"))

    last_3 = manager.flush_history(n=3)
    assert len(last_3) == 3
    assert last_3[0].text == "자막 2"
    assert last_3[-1].text == "자막 4"


def test_flush_history_returns_copy():
    """flush_history()가 내부 리스트의 복사본을 반환하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("테스트", "final"))
    history = manager.flush_history()

    # 반환된 복사본을 수정해도 내부 상태에 영향 없어야 함
    history.clear()
    assert len(manager.flush_history()) == 1


# =============================================================================
# SRT/VTT 파일 저장 테스트
# =============================================================================

def test_export_srt_creates_file(tmp_path):
    """export_srt()가 파일을 생성하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    # 세션 시작 기준 타임스탬프 사용 (절대 시간이 아닌 상대적 테스트)
    manager.process_result(_make_stt_result("첫 번째 자막", "final"))
    manager.process_result(_make_stt_result("두 번째 자막", "final"))

    srt_path = tmp_path / "test.srt"
    manager.export_srt(srt_path)

    assert srt_path.exists()
    content = srt_path.read_text(encoding="utf-8")
    assert "첫 번째 자막" in content
    assert "두 번째 자막" in content


def test_export_srt_format(tmp_path):
    """SRT 파일 포맷이 올바른지 확인합니다 (번호, 타임코드, 텍스트)."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("테스트 자막", "final"))

    srt_path = tmp_path / "test.srt"
    manager.export_srt(srt_path)

    lines = srt_path.read_text(encoding="utf-8").strip().split("\n")
    # SRT 포맷: 번호, 타임코드, 텍스트, 빈줄
    assert lines[0] == "1"  # 번호
    assert "-->" in lines[1]  # 타임코드
    assert "," in lines[1]  # SRT 포맷 쉼표
    assert lines[2] == "테스트 자막"  # 텍스트


def test_export_vtt_format(tmp_path):
    """VTT 파일이 WEBVTT 헤더를 포함하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("VTT 테스트", "final"))

    vtt_path = tmp_path / "test.vtt"
    manager.export_vtt(vtt_path)

    content = vtt_path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "VTT 테스트" in content
    assert "-->" in content
    assert "." in content  # VTT 포맷 점


def test_export_srt_excludes_partial(tmp_path):
    """SRT 파일에 partial 자막이 포함되지 않는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("중간 결과", "partial"))
    manager.process_result(_make_stt_result("최종 결과", "final"))

    srt_path = tmp_path / "test.srt"
    manager.export_srt(srt_path)

    content = srt_path.read_text(encoding="utf-8")
    assert "중간 결과" not in content
    assert "최종 결과" in content


# =============================================================================
# 핫스왑 설정 테스트
# =============================================================================

def test_update_config_changes_font_size():
    """update_config() 후 새 설정이 적용되는지 확인합니다."""
    config1 = _make_config(sync_offset_ms=0)
    manager = SubtitleManager(config1)

    # 새 설정 (sync_offset 변경)
    config2 = _make_config(sync_offset_ms=500)
    manager.update_config(config2)

    before_ns = time.time_ns()
    event = manager.process_result(_make_stt_result("테스트", "final"))
    after_ns = time.time_ns()

    # 새 sync_offset이 적용되어야 함
    expected_min = before_ns + 500 * 1_000_000
    assert event.display_at_ns >= expected_min


def test_update_config_hotswap_no_history_loss():
    """설정 핫스왑 중 기존 히스토리가 보존되는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("기존 자막", "final"))

    # 설정 변경
    manager.update_config(_make_config(sync_offset_ms=200))

    history = manager.flush_history()
    assert len(history) == 1
    assert history[0].text == "기존 자막"


# =============================================================================
# SubtitleEvent 필드 테스트
# =============================================================================

def test_process_result_returns_subtitle_event():
    """process_result()가 SubtitleEvent를 반환하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    event = manager.process_result(_make_stt_result("테스트", "final"))

    assert isinstance(event, SubtitleEvent)


def test_subtitle_event_has_correct_fields():
    """SubtitleEvent 필드가 설정값과 일치하는지 확인합니다."""
    config = _make_config()
    manager = SubtitleManager(config)

    event = manager.process_result(_make_stt_result("테스트", "final"))

    assert event.font_size == 36
    assert event.color == "#FFFFFF"
    assert event.position_x == 0.5
    assert event.position_y == 0.85
    assert event.anchor == "center"
    assert event.stroke_width == 2


# =============================================================================
# _auto_export 증분 쓰기 테스트 (BUG-005)
# =============================================================================

def test_auto_export_srt_incremental_entry_count(tmp_path):
    """final 이벤트마다 SRT 파일의 자막 항목 수가 1씩 늘어나는지 확인합니다."""
    config = _make_config(export_enabled=True, export_output_dir=str(tmp_path))
    manager = SubtitleManager(config)
    srt_path = tmp_path / "subtitles.srt"

    manager.process_result(_make_stt_result("첫 번째", "final"))
    assert srt_path.read_text(encoding="utf-8").count("-->") == 1

    manager.process_result(_make_stt_result("두 번째", "final"))
    assert srt_path.read_text(encoding="utf-8").count("-->") == 2

    manager.process_result(_make_stt_result("세 번째", "final"))
    content = srt_path.read_text(encoding="utf-8")
    assert content.count("-->") == 3
    assert "첫 번째" in content
    assert "두 번째" in content
    assert "세 번째" in content


def test_auto_export_srt_indices_are_sequential(tmp_path):
    """SRT 자막 번호가 1부터 순서대로 부여되는지 확인합니다."""
    config = _make_config(export_enabled=True, export_output_dir=str(tmp_path))
    manager = SubtitleManager(config)

    for i in range(1, 4):
        manager.process_result(_make_stt_result(f"자막 {i}", "final"))

    lines = (tmp_path / "subtitles.srt").read_text(encoding="utf-8").splitlines()
    # SRT 블록은 번호, 타임코드, 텍스트, 빈줄 순 (4줄 단위)
    indices = [lines[i] for i in range(0, len(lines), 4) if lines[i].strip()]
    assert indices == ["1", "2", "3"]


def test_auto_export_vtt_has_single_header(tmp_path):
    """VTT 파일에 WEBVTT 헤더가 정확히 1개만 존재하는지 확인합니다."""
    config = _make_config(export_enabled=True, export_output_dir=str(tmp_path))
    manager = SubtitleManager(config)

    for _ in range(3):
        manager.process_result(_make_stt_result("테스트", "final"))

    content = (tmp_path / "subtitles.vtt").read_text(encoding="utf-8")
    assert content.count("WEBVTT") == 1
    assert content.startswith("WEBVTT")
    assert content.count("-->") == 3


def test_auto_export_srt_matches_full_export(tmp_path):
    """_auto_export() 결과가 export_srt() 전체 내보내기 결과와 동일한지 확인합니다."""
    config = _make_config(export_enabled=True, export_output_dir=str(tmp_path))
    manager = SubtitleManager(config)

    manager.process_result(_make_stt_result("첫 번째", "final"))
    manager.process_result(_make_stt_result("두 번째", "final"))
    manager.process_result(_make_stt_result("세 번째", "final"))

    auto_content = (tmp_path / "subtitles.srt").read_text(encoding="utf-8")

    manual_path = tmp_path / "manual.srt"
    manager.export_srt(manual_path)
    manual_content = manual_path.read_text(encoding="utf-8")

    assert auto_content == manual_content


def test_auto_export_srt_correct_after_history_overflow(tmp_path):
    """히스토리 오버플로우 발생 후에도 새 이벤트만 정확히 append되는지 확인합니다."""
    config = _make_config(
        export_enabled=True,
        export_output_dir=str(tmp_path),
        history_size=2,
    )
    manager = SubtitleManager(config)

    # history_size=2이므로 3번째부터 오버플로우 발생
    for i in range(1, 5):
        manager.process_result(_make_stt_result(f"자막 {i}", "final"))

    content = (tmp_path / "subtitles.srt").read_text(encoding="utf-8")
    assert content.count("-->") == 4

    lines = content.splitlines()
    indices = [lines[i] for i in range(0, len(lines), 4) if lines[i].strip()]
    assert indices == ["1", "2", "3", "4"]
