"""
자막 모듈 패키지

공통 데이터 타입:
- SubtitleEvent: 자막 표시 이벤트 컨테이너
"""

from dataclasses import dataclass, field


@dataclass
class SubtitleEvent:
    """
    자막 표시 이벤트 컨테이너입니다.

    VideoCompositor가 이 이벤트를 참고하여 비디오 프레임에 자막을 오버레이합니다.

    필드:
        event_id: 이벤트 순번 (0부터 시작)
        text: 표시할 자막 텍스트
        is_partial: True이면 partial(중간) 결과, False이면 final(확정) 결과
        display_at_ns: 자막 표시 시작 시각 (nanoseconds, sync_offset 적용 후)
        expire_at_ns: 자막 만료 시각 (nanoseconds, display_at_ns + display_duration)
        font_path: 폰트 파일 경로
        font_size: 폰트 크기 (픽셀)
        color: 텍스트 색상 (HEX, 예: "#FFFFFF")
        background_color: 배경 색상 (HEX + 투명도, 예: "#00000080")
        stroke_color: 외곽선 색상
        stroke_width: 외곽선 두께
        position_x: 가로 위치 (0.0~1.0)
        position_y: 세로 위치 (0.0~1.0)
        anchor: 텍스트 정렬 ("center" | "left" | "right")
    """
    event_id: int
    text: str
    is_partial: bool
    display_at_ns: int
    expire_at_ns: int
    font_path: str
    font_size: int
    color: str
    background_color: str
    stroke_color: str
    stroke_width: int
    position_x: float
    position_y: float
    anchor: str
