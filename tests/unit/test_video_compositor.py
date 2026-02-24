"""
VideoCompositor 단위 테스트

검증 항목:
- YUV422 VideoFrame → numpy BGR 변환
- 자막 오버레이 적용 (픽셀 변화 확인)
- 프레임 드롭 감지 (frame_id gap)
- BufferStatus 반환
- HEX 색상 변환 유틸리티
- 핫스왑 설정 업데이트
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.capture import VideoFrame
from src.compositor import BufferStatus
from src.compositor.video_compositor import VideoCompositor, _hex_to_rgb, _hex_to_rgba
from src.config.schema import AppConfig
from src.subtitle import SubtitleEvent


# =============================================================================
# 테스트 헬퍼
# =============================================================================

def _make_config() -> AppConfig:
    """테스트용 AppConfig를 생성합니다."""
    return AppConfig(**{
        "capture": {
            "video_queue_size": 30,
        },
        "subtitle": {
            "font": {
                "path": "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "size": 36,
                "color": "#FFFFFF",
                "background_color": "#00000080",
                "stroke_color": "#000000",
                "stroke_width": 2,
            },
            "position": {"x": 0.5, "y": 0.85, "anchor": "center"},
        },
    })


def _make_black_uyvy_frame(width: int = 320, height: int = 240) -> VideoFrame:
    """
    테스트용 검정 YUV422 UYVY VideoFrame을 생성합니다.

    실제 해상도(1920x1080) 대신 소형 해상도를 사용하여 테스트 속도를 높입니다.
    """
    total_macro_pixels = (width * height) // 2
    frame_array = np.empty(total_macro_pixels * 4, dtype=np.uint8)
    frame_array[0::4] = 128  # U
    frame_array[1::4] = 16   # Y0 (검정)
    frame_array[2::4] = 128  # V
    frame_array[3::4] = 16   # Y1 (검정)
    return VideoFrame(
        frame_id=0,
        timestamp_ns=1_000_000_000,
        width=width,
        height=height,
        pixel_format="yuv422",
        data=frame_array.tobytes(),
    )


def _make_bgra_frame(width: int = 320, height: int = 240) -> VideoFrame:
    """테스트용 BGRA VideoFrame을 생성합니다."""
    data = np.full((height, width, 4), fill_value=[128, 64, 32, 255], dtype=np.uint8)
    return VideoFrame(
        frame_id=0,
        timestamp_ns=1_000_000_000,
        width=width,
        height=height,
        pixel_format="bgra",
        data=data.tobytes(),
    )


def _make_subtitle_event(text: str = "테스트 자막") -> SubtitleEvent:
    """테스트용 SubtitleEvent를 생성합니다."""
    return SubtitleEvent(
        event_id=0,
        text=text,
        is_partial=False,
        display_at_ns=0,
        expire_at_ns=int(1e18),  # 매우 먼 미래
        font_path="/System/Library/Fonts/AppleSDGothicNeo.ttc",
        font_size=24,
        color="#FFFFFF",
        background_color="#00000080",
        stroke_color="#000000",
        stroke_width=1,
        position_x=0.5,
        position_y=0.85,
        anchor="center",
    )


# =============================================================================
# 프레임 변환 테스트
# =============================================================================

def test_composite_yuv422_returns_bgr_array():
    """YUV422 프레임이 BGR numpy 배열로 변환되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)
    frame = _make_black_uyvy_frame(width=320, height=240)

    result = compositor.composite(frame, None)

    assert isinstance(result, np.ndarray)
    assert result.shape == (240, 320, 3)
    assert result.dtype == np.uint8


def test_composite_bgra_returns_bgr_array():
    """BGRA 프레임이 BGR numpy 배열로 변환되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)
    frame = _make_bgra_frame(width=320, height=240)

    result = compositor.composite(frame, None)

    assert isinstance(result, np.ndarray)
    assert result.shape == (240, 320, 3)


def test_composite_unknown_format_returns_black_frame():
    """알 수 없는 픽셀 포맷은 검정 배경을 반환하는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)
    frame = VideoFrame(
        frame_id=0,
        timestamp_ns=0,
        width=320,
        height=240,
        pixel_format="unknown_format",
        data=b"\x00" * (320 * 240 * 3),
    )

    result = compositor.composite(frame, None)

    assert result.shape == (240, 320, 3)
    assert result.sum() == 0  # 검정 배경


# =============================================================================
# 자막 오버레이 테스트
# =============================================================================

def test_composite_with_subtitle_changes_pixels():
    """자막 오버레이 적용 시 픽셀 값이 변경되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)
    frame = _make_black_uyvy_frame(width=640, height=480)
    event = _make_subtitle_event("Hello")

    without_subtitle = compositor.composite(frame, None).copy()

    # expected_frame_id를 리셋하여 드롭 감지 방지
    compositor._expected_frame_id = 0
    with_subtitle = compositor.composite(frame, event)

    # 자막이 없는 프레임과 있는 프레임은 달라야 함
    assert not np.array_equal(without_subtitle, with_subtitle), \
        "자막 오버레이 후 픽셀 변화 없음"


def test_composite_empty_subtitle_no_overlay():
    """빈 텍스트 자막은 오버레이가 적용되지 않는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)
    frame = _make_black_uyvy_frame(width=320, height=240)

    event = _make_subtitle_event(text="")  # 빈 텍스트
    result = compositor.composite(frame, event)
    without = compositor.composite(frame, None)
    compositor._expected_frame_id = 0

    # 빈 텍스트는 오버레이 없으므로 두 결과가 같아야 함
    # (픽셀이 완전히 동일하지 않을 수 있으므로 근사 비교)
    diff_pixels = np.sum(result != without)
    # 빈 텍스트로 인한 픽셀 변화가 거의 없어야 함
    assert diff_pixels < 100, f"빈 텍스트인데 {diff_pixels}개 픽셀 변화"


# =============================================================================
# 프레임 드롭 감지 테스트
# =============================================================================

def test_detect_frame_drop_consecutive_no_drop():
    """연속된 frame_id는 드롭으로 감지되지 않는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)

    dropped = compositor.detect_frame_drop(frame_id=1, expected_id=1)
    assert dropped is False
    assert compositor._drop_count == 0


def test_detect_frame_drop_gap_detected():
    """frame_id에 gap이 있으면 드롭으로 감지되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)

    dropped = compositor.detect_frame_drop(frame_id=5, expected_id=3)
    assert dropped is True
    assert compositor._drop_count == 2  # gap = 5 - 3 = 2


def test_detect_frame_drop_accumulates():
    """여러 번 드롭이 발생하면 _drop_count가 누적되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)

    compositor.detect_frame_drop(frame_id=2, expected_id=0)  # gap=2
    compositor.detect_frame_drop(frame_id=7, expected_id=3)  # gap=4

    assert compositor._drop_count == 6


def test_composite_updates_expected_frame_id():
    """composite() 호출 후 _expected_frame_id가 갱신되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)

    frame = _make_black_uyvy_frame()
    frame_modified = VideoFrame(
        frame_id=5,
        timestamp_ns=frame.timestamp_ns,
        width=frame.width,
        height=frame.height,
        pixel_format=frame.pixel_format,
        data=frame.data,
    )

    compositor.composite(frame_modified, None)

    assert compositor._expected_frame_id == 6


# =============================================================================
# BufferStatus 테스트
# =============================================================================

def test_get_buffer_status_no_queue():
    """video_queue가 없을 때 BufferStatus가 올바른지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config, video_queue=None)

    status = compositor.get_buffer_status()

    assert isinstance(status, BufferStatus)
    assert status.queue_depth == 0
    assert status.max_depth == 30
    assert status.drop_count == 0
    assert status.drop_rate == 0.0


def test_get_buffer_status_with_queue():
    """video_queue가 있을 때 큐 깊이가 반영되는지 확인합니다."""
    config = _make_config()
    queue = asyncio.Queue(maxsize=30)
    # 큐에 5개 항목 추가
    for _ in range(5):
        queue.put_nowait(None)

    compositor = VideoCompositor(config, video_queue=queue)
    status = compositor.get_buffer_status()

    assert status.queue_depth == 5
    assert status.max_depth == 30


def test_drop_rate_calculation():
    """드롭율이 올바르게 계산되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)

    # 10프레임 처리 중 2프레임 드롭 시뮬레이션
    compositor._total_frames = 10
    compositor._drop_count = 2

    status = compositor.get_buffer_status()

    assert abs(status.drop_rate - 0.2) < 0.001


# =============================================================================
# 핫스왑 설정 테스트
# =============================================================================

def test_update_style_clears_font_cache():
    """update_style() 후 폰트 캐시가 초기화되는지 확인합니다."""
    config = _make_config()
    compositor = VideoCompositor(config)

    # 폰트 캐시에 더미 항목 추가
    compositor._font_cache["dummy:36"] = "cached_font"

    compositor.update_style(config)

    assert len(compositor._font_cache) == 0


# =============================================================================
# 색상 변환 테스트
# =============================================================================

def test_hex_to_rgb_white():
    """#FFFFFF가 (255, 255, 255)로 변환되는지 확인합니다."""
    assert _hex_to_rgb("#FFFFFF") == (255, 255, 255)


def test_hex_to_rgb_black():
    """#000000이 (0, 0, 0)으로 변환되는지 확인합니다."""
    assert _hex_to_rgb("#000000") == (0, 0, 0)


def test_hex_to_rgb_custom():
    """#FF8040이 (255, 128, 64)로 변환되는지 확인합니다."""
    assert _hex_to_rgb("#FF8040") == (255, 128, 64)


def test_hex_to_rgba_with_alpha():
    """#00000080이 (0, 0, 0, 128)로 변환되는지 확인합니다."""
    assert _hex_to_rgba("#00000080") == (0, 0, 0, 128)


def test_hex_to_rgba_no_alpha():
    """6자리 HEX는 알파=255로 변환되는지 확인합니다."""
    r, g, b, a = _hex_to_rgba("#FFFFFF")
    assert a == 255


# =============================================================================
# stroke 렌더링 최적화 테스트 (BUG-006)
# =============================================================================

def test_draw_text_uses_pillow_stroke_param(monkeypatch):
    """stroke 렌더링 시 Pillow stroke_width 파라미터로 draw.text()가 1회만 호출되는지 확인합니다."""
    _ImageDraw = pytest.importorskip("PIL.ImageDraw", reason="Pillow not installed")

    draw_text_calls = []
    original_text = _ImageDraw.ImageDraw.text

    def patched_text(self, xy, text, fill=None, font=None, **kwargs):
        draw_text_calls.append(kwargs)
        return original_text(self, xy, text, fill=fill, font=font, **kwargs)

    monkeypatch.setattr(_ImageDraw.ImageDraw, "text", patched_text)

    config = _make_config()
    compositor = VideoCompositor(config)
    frame = _make_black_uyvy_frame(width=320, height=240)
    event = SubtitleEvent(
        event_id=0,
        text="Hello",
        is_partial=False,
        display_at_ns=0,
        expire_at_ns=int(1e18),
        font_path="/System/Library/Fonts/AppleSDGothicNeo.ttc",
        font_size=24,
        color="#FFFFFF",
        background_color="#00000080",
        stroke_color="#000000",
        stroke_width=3,  # 루프 구현이었다면 49회 호출됐을 값
        position_x=0.5,
        position_y=0.85,
        anchor="center",
    )

    bgr_frame = compositor._frame_to_bgr(frame)
    compositor._draw_text_pil(bgr_frame, event)

    # stroke_width=3이어도 draw.text()는 정확히 1번만 호출
    assert len(draw_text_calls) == 1
    # Pillow 내장 stroke 파라미터가 전달되어야 함
    assert draw_text_calls[0].get("stroke_width") == 3
    assert draw_text_calls[0].get("stroke_fill") is not None
