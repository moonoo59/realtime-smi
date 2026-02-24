"""
STT 모듈 패키지

공통 데이터 타입:
- STTResult: STT 인식 결과 컨테이너 (partial/final)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WordTiming:
    """단어 단위 타임스탬프입니다."""
    word: str
    start_ms: int
    end_ms: int


@dataclass
class STTResult:
    """
    STT 인식 결과 컨테이너입니다.

    필드:
        result_id: 결과 순번 (0부터 시작)
        type: 결과 유형 ("partial" | "final")
        text: 인식된 텍스트
        send_timestamp_ns: 오디오 청크 전송 시각 (nanoseconds)
        receive_timestamp_ns: STT 결과 수신 시각 (nanoseconds)
        confidence: 신뢰도 (0.0~1.0)
        words: 단어 단위 타임스탬프 목록
    """
    result_id: int
    type: Literal["partial", "final"]
    text: str
    send_timestamp_ns: int
    receive_timestamp_ns: int
    confidence: float
    words: list[WordTiming] = field(default_factory=list)
    last_chunk_id: int = 0  # 마지막으로 전송된 PCMChunk.chunk_id (LatencyTracker 연동용)
