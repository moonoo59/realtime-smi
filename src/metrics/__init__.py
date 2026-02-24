"""
메트릭 모듈 패키지

공통 데이터 타입:
- LatencyStats: 지연시간 통계 컨테이너
- AccuracyReport: WER/CER 정확도 리포트
"""

from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """
    파이프라인 단계별 지연시간 통계입니다.

    필드:
        stage: 파이프라인 단계 이름
        count: 측정 샘플 수
        mean_ms: 평균 지연시간 (밀리초)
        min_ms: 최소 지연시간 (밀리초)
        max_ms: 최대 지연시간 (밀리초)
        p95_ms: 95th percentile 지연시간 (밀리초)
        p99_ms: 99th percentile 지연시간 (밀리초)
    """
    stage: str
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class ErrorDetail:
    """WER/CER 오류 상세 정보입니다."""
    hypothesis: str
    reference: str
    wer: float


@dataclass
class AccuracyReport:
    """
    WER/CER 정확도 리포트입니다.

    필드:
        session_id: 세션 식별자
        duration_sec: 세션 총 시간 (초)
        total_words: 총 단어 수
        total_chars: 총 문자 수
        wer: 단어 오류율 (0.0~1.0)
        cer: 문자 오류율 (0.0~1.0)
        error_details: 개별 발화 오류 상세 목록
    """
    session_id: str
    duration_sec: float
    total_words: int
    total_chars: int
    wer: float
    cer: float
    error_details: list[ErrorDetail] = field(default_factory=list)
