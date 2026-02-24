"""
비디오 컴포지터 모듈 패키지

공통 데이터 타입:
- BufferStatus: 비디오 큐 상태 정보
"""

from dataclasses import dataclass


@dataclass
class BufferStatus:
    """
    비디오 큐 버퍼 상태 정보입니다.

    필드:
        queue_depth: 현재 큐 항목 수
        max_depth: 큐 최대 크기
        drop_count: 누적 프레임 드롭 횟수
        drop_rate: 프레임 드롭율 (0.0~1.0)
    """
    queue_depth: int
    max_depth: int
    drop_count: int
    drop_rate: float
