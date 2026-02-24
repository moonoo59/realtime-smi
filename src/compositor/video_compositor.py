"""
비디오 컴포지터 모듈입니다.

역할:
- 비디오 프레임(YUV422/BGRA bytes)을 numpy 배열로 변환
- Pillow 기반 한글 자막 텍스트 렌더링
- 반투명 배경 박스 오버레이
- OpenCV imshow로 프리뷰 출력
- 프레임 드롭 감지 (frame_id 기반)
- 비디오 큐 깊이 및 드롭율 모니터링

사용 예시:
    >>> compositor = VideoCompositor(config, video_queue)
    >>> frame_np = compositor.composite(video_frame, subtitle_event)
    >>> compositor.display(frame_np)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.capture import VideoFrame
from src.compositor import BufferStatus
from src.config.schema import AppConfig
from src.subtitle import SubtitleEvent

logger = logging.getLogger(__name__)

# Pillow 한글 렌더링 (선택적 의존성)
try:
    from PIL import Image, ImageDraw, ImageFont
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False
    logger.warning("Pillow가 설치되지 않아 기본 OpenCV 폰트를 사용합니다.")


class VideoCompositor:
    """
    비디오 프레임에 자막을 오버레이하고 화면에 출력하는 클래스입니다.

    Pillow로 한글 텍스트를 렌더링하고, OpenCV imshow로 프리뷰를 출력합니다.
    Pillow가 없는 경우 OpenCV 기본 폰트로 폴백합니다.

    프레임 드롭 감지:
        연속적으로 수신되어야 하는 frame_id를 추적하여
        gap이 발생하면 드롭으로 카운트합니다.
    """

    # 미리보기 창 이름
    _WINDOW_NAME = "SDI-RealtimeSubtitle Preview"

    def __init__(
        self,
        config: AppConfig,
        video_queue: Optional[asyncio.Queue] = None,
    ) -> None:
        """
        VideoCompositor를 초기화합니다.

        파라미터:
            config (AppConfig): 전체 애플리케이션 설정 객체
            video_queue (Optional[asyncio.Queue]): 버퍼 상태 모니터링용 큐
        """
        self._config = config
        self._compositor_cfg = config.subtitle  # 폰트/위치 설정은 subtitle 섹션 사용
        self._capture_cfg = config.capture
        self._video_queue = video_queue

        # 프레임 드롭 추적
        self._expected_frame_id: int = 0
        self._drop_count: int = 0
        self._total_frames: int = 0

        # Pillow 폰트 캐시 (폰트 경로+크기 → ImageFont 인스턴스)
        self._font_cache: dict[str, ImageFont.FreeTypeFont] = {}

        logger.info("VideoCompositor 초기화 완료")

    # =========================================================================
    # 공개 인터페이스
    # =========================================================================

    def composite(
        self,
        video_frame: VideoFrame,
        subtitle_event: Optional[SubtitleEvent] = None,
    ) -> np.ndarray:
        """
        비디오 프레임에 자막을 오버레이하여 numpy 배열을 반환합니다.

        처리 순서:
        1. VideoFrame bytes → numpy BGR 배열
        2. 자막 이벤트가 있으면 텍스트 오버레이 적용
        3. 프레임 드롭 감지

        파라미터:
            video_frame (VideoFrame): 원본 비디오 프레임
            subtitle_event (Optional[SubtitleEvent]): 표시할 자막 (None이면 오버레이 없음)

        반환값:
            np.ndarray: BGR 포맷 numpy 배열
        """
        # 프레임 드롭 감지
        self.detect_frame_drop(video_frame.frame_id, self._expected_frame_id)
        self._expected_frame_id = video_frame.frame_id + 1
        self._total_frames += 1

        # bytes → numpy BGR
        bgr_frame = self._frame_to_bgr(video_frame)

        # 자막 오버레이
        if subtitle_event is not None and subtitle_event.text.strip():
            bgr_frame = self._draw_subtitle(bgr_frame, subtitle_event)

        return bgr_frame

    def display(self, frame: np.ndarray) -> bool:
        """
        numpy 배열을 OpenCV 미리보기 창으로 출력합니다.

        파라미터:
            frame: BGR 포맷 numpy 배열

        반환값:
            bool: 창이 닫히지 않은 경우 True, 'q' 키 입력 또는 창 닫힘이면 False
        """
        try:
            cv2.imshow(self._WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            return True
        except Exception as exc:
            logger.error(f"OpenCV 화면 출력 실패: {exc}")
            return False

    def detect_frame_drop(self, frame_id: int, expected_id: int) -> bool:
        """
        현재 frame_id와 예상 frame_id를 비교하여 드롭 여부를 반환합니다.

        파라미터:
            frame_id: 현재 프레임 ID
            expected_id: 예상 프레임 ID

        반환값:
            bool: 프레임 드롭이 발생했으면 True
        """
        dropped = frame_id > expected_id
        if dropped:
            drop_gap = frame_id - expected_id
            self._drop_count += drop_gap
            logger.warning(
                f"프레임 드롭 감지: frame_id={frame_id}, "
                f"expected={expected_id}, gap={drop_gap}"
            )
        return dropped

    def get_buffer_status(self) -> BufferStatus:
        """
        비디오 큐 깊이 및 드롭 통계를 반환합니다.

        반환값:
            BufferStatus: 현재 버퍼 상태
        """
        if self._video_queue is not None:
            queue_depth = self._video_queue.qsize()
            max_depth = self._capture_cfg.video_queue_size
        else:
            queue_depth = 0
            max_depth = self._capture_cfg.video_queue_size

        drop_rate = (
            self._drop_count / self._total_frames
            if self._total_frames > 0
            else 0.0
        )

        return BufferStatus(
            queue_depth=queue_depth,
            max_depth=max_depth,
            drop_count=self._drop_count,
            drop_rate=drop_rate,
        )

    def update_style(self, config: AppConfig) -> None:
        """
        폰트/색상/위치 설정을 핫스왑으로 업데이트합니다.

        파라미터:
            config (AppConfig): 새 설정 객체
        """
        self._compositor_cfg = config.subtitle
        self._font_cache.clear()  # 폰트 캐시 초기화
        logger.info(
            f"VideoCompositor 설정 핫스왑: "
            f"font_size={config.subtitle.font.size}"
        )

    def close(self) -> None:
        """OpenCV 창을 닫습니다."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # =========================================================================
    # 내부 렌더링 메서드
    # =========================================================================

    def _frame_to_bgr(self, frame: VideoFrame) -> np.ndarray:
        """
        VideoFrame bytes를 OpenCV BGR numpy 배열로 변환합니다.

        지원 픽셀 포맷:
        - yuv422 (UYVY): cv2.COLOR_YUV2BGR_UYVY
        - bgra: 4채널 → 3채널 변환
        - 기타: 검정 배경 생성

        파라미터:
            frame: 변환할 VideoFrame

        반환값:
            np.ndarray: BGR 포맷 배열 (height, width, 3)
        """
        try:
            if frame.pixel_format == "yuv422":
                return self._uyvy_to_bgr(frame)
            elif frame.pixel_format == "bgra":
                return self._bgra_to_bgr(frame)
            else:
                logger.warning(f"지원하지 않는 픽셀 포맷: {frame.pixel_format}")
                return np.zeros((frame.height, frame.width, 3), dtype=np.uint8)

        except Exception as exc:
            logger.error(f"프레임 변환 실패: {exc}", exc_info=True)
            return np.zeros((frame.height, frame.width, 3), dtype=np.uint8)

    def _uyvy_to_bgr(self, frame: VideoFrame) -> np.ndarray:
        """
        YUV422 UYVY 포맷을 BGR로 변환합니다.

        UYVY 레이아웃: 2픽셀당 4바이트 [U0, Y0, V0, Y1]
        OpenCV COLOR_YUV2BGR_UYVY는 (H, W, 2) 2채널 입력을 요구합니다.

        파라미터:
            frame: UYVY 포맷 VideoFrame

        반환값:
            np.ndarray: BGR 배열
        """
        raw = np.frombuffer(frame.data, dtype=np.uint8)
        # (H × W × 2) → (H, W, 2): 각 픽셀당 2바이트 [U/V, Y] 2채널
        uyvy = raw.reshape(frame.height, frame.width, 2)
        bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
        return bgr

    def _bgra_to_bgr(self, frame: VideoFrame) -> np.ndarray:
        """
        BGRA 포맷을 BGR로 변환합니다.

        파라미터:
            frame: BGRA 포맷 VideoFrame

        반환값:
            np.ndarray: BGR 배열
        """
        raw = np.frombuffer(frame.data, dtype=np.uint8)
        bgra = raw.reshape(frame.height, frame.width, 4)
        return bgra[:, :, :3]

    def _draw_subtitle(self, frame: np.ndarray, event: SubtitleEvent) -> np.ndarray:
        """
        자막 텍스트를 프레임에 오버레이합니다.

        Pillow 사용 가능 시: 한글 폰트 렌더링 + 반투명 배경
        Pillow 미설치 시: OpenCV 기본 폰트로 폴백

        파라미터:
            frame: BGR 배열
            event: 자막 이벤트

        반환값:
            np.ndarray: 자막이 오버레이된 BGR 배열
        """
        if _PILLOW_AVAILABLE:
            return self._draw_text_pil(frame, event)
        else:
            return self._draw_text_opencv(frame, event)

    def _draw_text_pil(self, frame: np.ndarray, event: SubtitleEvent) -> np.ndarray:
        """
        Pillow를 사용하여 한글 자막 텍스트를 렌더링합니다.

        처리 순서:
        1. BGR → PIL Image 변환
        2. 텍스트 크기 측정
        3. 반투명 배경 박스 그리기
        4. 텍스트 렌더링 (외곽선 + 본문)
        5. PIL Image → BGR 변환

        파라미터:
            frame: BGR 배열
            event: 자막 이벤트

        반환값:
            np.ndarray: 자막이 렌더링된 BGR 배열
        """
        height, width = frame.shape[:2]

        # BGR → PIL RGB
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, "RGBA")

        # 폰트 로드 (캐시 사용)
        font = self._load_font(event.font_path, event.font_size)

        # 텍스트 표시 위치 계산
        text_x, text_y = self._calculate_text_position(
            draw, event.text, font, width, height, event
        )

        # 배경 박스 그리기
        self._draw_background_box(draw, event.text, font, text_x, text_y, event)

        # 텍스트 및 외곽선 그리기 (Pillow 내장 stroke 파라미터 사용 — O(1))
        stroke_color = _hex_to_rgb(event.stroke_color)
        text_color = _hex_to_rgb(event.color) + (255,)
        draw.text(
            (text_x, text_y),
            event.text,
            font=font,
            fill=text_color,
            stroke_width=event.stroke_width,
            stroke_fill=stroke_color + (255,),
        )

        # PIL RGB → BGR
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _draw_text_opencv(self, frame: np.ndarray, event: SubtitleEvent) -> np.ndarray:
        """
        Pillow 없이 OpenCV 기본 폰트로 자막을 렌더링합니다.

        한글이 깨질 수 있으나 Pillow 미설치 환경의 폴백으로 사용합니다.

        파라미터:
            frame: BGR 배열
            event: 자막 이벤트

        반환값:
            np.ndarray: 자막이 렌더링된 BGR 배열
        """
        height, width = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = event.font_size / 30.0
        thickness = max(1, event.stroke_width)

        text_size, _ = cv2.getTextSize(event.text, font, font_scale, thickness)
        text_w, text_h = text_size

        # 위치 계산
        x = int(width * event.position_x - text_w / 2)
        y = int(height * event.position_y)
        x = max(0, min(x, width - text_w))
        y = max(text_h, min(y, height))

        # 텍스트 색상 (BGR 순서)
        color_rgb = _hex_to_rgb(event.color)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.putText(frame, event.text, (x, y), font, font_scale, color_bgr, thickness)
        return frame

    def _load_font(
        self, font_path: str, font_size: int
    ) -> ImageFont.FreeTypeFont:
        """
        폰트를 로드합니다. 캐시에 없으면 새로 로드하여 캐시에 저장합니다.

        폰트 파일이 없으면 기본 폰트로 폴백합니다.

        파라미터:
            font_path: 폰트 파일 경로
            font_size: 폰트 크기

        반환값:
            ImageFont: 로드된 폰트
        """
        cache_key = f"{font_path}:{font_size}"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        try:
            font = ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            logger.warning(
                f"폰트 파일을 찾을 수 없습니다: {font_path}. "
                "기본 폰트로 폴백합니다 (한글 깨짐 가능)."
            )
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def _calculate_text_position(
        self,
        draw: ImageDraw.Draw,
        text: str,
        font: ImageFont.FreeTypeFont,
        frame_width: int,
        frame_height: int,
        event: SubtitleEvent,
    ) -> tuple[int, int]:
        """
        자막 텍스트의 렌더링 위치를 계산합니다.

        파라미터:
            draw: PIL ImageDraw 인스턴스
            text: 자막 텍스트
            font: 폰트
            frame_width, frame_height: 프레임 크기
            event: 자막 이벤트 (position_x, position_y, anchor 포함)

        반환값:
            tuple[int, int]: (x, y) 픽셀 좌표
        """
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # anchor 기반 x 위치 계산
        center_x = int(frame_width * event.position_x)
        if event.anchor == "center":
            x = center_x - text_w // 2
        elif event.anchor == "right":
            x = center_x - text_w
        else:  # left
            x = center_x

        y = int(frame_height * event.position_y) - text_h

        # 화면 경계 클리핑
        x = max(0, min(x, frame_width - text_w))
        y = max(0, min(y, frame_height - text_h))

        return x, y

    def _draw_background_box(
        self,
        draw: ImageDraw.Draw,
        text: str,
        font: ImageFont.FreeTypeFont,
        text_x: int,
        text_y: int,
        event: SubtitleEvent,
        padding: int = 8,
    ) -> None:
        """
        자막 텍스트 아래에 반투명 배경 박스를 그립니다.

        파라미터:
            draw: PIL ImageDraw 인스턴스
            text: 자막 텍스트
            font: 폰트
            text_x, text_y: 텍스트 위치
            event: 자막 이벤트 (background_color 포함)
            padding: 배경 박스 패딩 (픽셀)
        """
        bbox = draw.textbbox((text_x, text_y), text, font=font)

        bg_box = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        )

        # HEX + 투명도 → RGBA
        bg_color = _hex_to_rgba(event.background_color)
        draw.rectangle(bg_box, fill=bg_color)


# =============================================================================
# 색상 변환 헬퍼
# =============================================================================

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    HEX 색상 문자열을 RGB 튜플로 변환합니다.

    파라미터:
        hex_color: HEX 색상 (예: "#FFFFFF" 또는 "#FFFFFFFF")

    반환값:
        tuple[int, int, int]: (R, G, B)
    """
    hex_color = hex_color.lstrip("#")
    # 투명도 포함 8자리인 경우 앞 6자리만 사용
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """
    HEX 색상 문자열(투명도 포함)을 RGBA 튜플로 변환합니다.

    파라미터:
        hex_color: HEX 색상 (예: "#00000080" = 검정 50% 투명)

    반환값:
        tuple[int, int, int, int]: (R, G, B, A)
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = _hex_to_rgb("#" + hex_color[:6])

    if len(hex_color) == 8:
        a = int(hex_color[6:8], 16)
    else:
        a = 255

    return (r, g, b, a)
