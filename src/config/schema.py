"""
SDI-RealtimeSubtitle 설정 스키마 정의 모듈입니다.

역할:
- Pydantic v2 BaseModel 기반으로 config.yaml의 전체 구조를 타입 안전하게 정의
- 각 섹션(system, capture, audio, stt, subtitle, accuracy, metrics, dashboard, alerts)을
  독립적인 중첩 모델로 분리하여 유지보수성 확보
- 필드별 기본값, 허용 범위, 유효성 검증(validator)을 포함

사용 예시:
    >>> from src.config.schema import AppConfig
    >>> config = AppConfig(**yaml_data)
    >>> print(config.stt.endpoint)
"""

from __future__ import annotations

import logging
from typing import Union

from pydantic import BaseModel, Field, field_validator

# 모듈 로거 설정
logger = logging.getLogger(__name__)


# =============================================================================
# system 섹션: 시스템 전역 설정
# =============================================================================

class SystemConfig(BaseModel):
    """
    시스템 전역 설정을 정의하는 모델입니다.

    역할:
    - 실행 모드(live/file) 결정
    - 로깅 레벨 및 포맷 지정
    - 세션 식별자 관리
    """
    # 실행 모드: "live"는 실제 SDI 캡처, "file"은 테스트용 파일 시뮬레이션
    mode: str = Field(default="live", description="실행 모드 (live | file)")
    # 로그 출력 레벨
    log_level: str = Field(default="INFO", description="로그 레벨 (DEBUG | INFO | WARNING | ERROR)")
    # 로그 출력 포맷
    log_format: str = Field(default="json", description="로그 포맷 (json | text)")
    # 로그 파일 저장 디렉토리 경로
    log_dir: str = Field(default="output/logs", description="로그 저장 디렉토리")
    # 세션 고유 식별자 (빈 문자열이면 UUID로 자동 생성)
    session_id: str = Field(default="", description="세션 ID (비어있으면 UUID 자동생성)")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """실행 모드가 허용된 값인지 검증합니다."""
        allowed_modes = ("live", "file")
        if value not in allowed_modes:
            error_message = f"mode는 {allowed_modes} 중 하나여야 합니다. 입력값: '{value}'"
            raise ValueError(error_message)
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """로그 레벨이 유효한 Python 로깅 레벨인지 검증합니다."""
        allowed_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        # 대소문자 구분 없이 비교 후 대문자로 정규화
        upper_value = value.upper()
        if upper_value not in allowed_levels:
            error_message = f"log_level은 {allowed_levels} 중 하나여야 합니다. 입력값: '{value}'"
            raise ValueError(error_message)
        return upper_value

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, value: str) -> str:
        """로그 포맷이 지원되는 형식인지 검증합니다."""
        allowed_formats = ("json", "text")
        if value not in allowed_formats:
            error_message = f"log_format은 {allowed_formats} 중 하나여야 합니다. 입력값: '{value}'"
            raise ValueError(error_message)
        return value


# =============================================================================
# capture 섹션: SDI 캡처 관련 설정
# =============================================================================

class TestFileConfig(BaseModel):
    """
    테스트 모드(mode=file)에서 사용할 파일 경로 설정입니다.

    역할:
    - 오디오/비디오 시뮬레이션용 파일 경로 지정
    - 반복 재생 및 재생 속도 제어
    """
    # 테스트용 오디오 파일 경로 (WAV/MP3)
    audio_path: str = Field(default="tests/fixtures/sample_audio.wav", description="테스트 오디오 파일 경로")
    # 테스트용 비디오 파일 경로 (비어있으면 검정 배경 자동 생성)
    video_path: str = Field(default="", description="테스트 비디오 파일 경로 (비어있으면 검정 배경)")
    # 파일 반복 재생 여부
    loop: bool = Field(default=False, description="파일 반복 재생 여부")
    # 재생 속도 배율 (1.0 = 실시간)
    playback_speed: float = Field(default=1.0, description="재생 속도 (1.0 = 실시간)")


class CaptureConfig(BaseModel):
    """
    SDI 캡처 장치 및 입력 설정을 정의하는 모델입니다.

    역할:
    - DeckLink 장치 인덱스 및 비디오 모드 설정
    - 오디오 채널/샘플레이트/비트뎁스 지정
    - 큐 크기 제한으로 메모리 사용량 제어
    - 테스트 모드용 파일 설정 포함
    """
    # DeckLink 장치 인덱스 (0부터 시작)
    device_index: int = Field(default=0, description="DeckLink 장치 인덱스")
    # 비디오 입력 모드
    video_mode: str = Field(default="1080p30", description="비디오 모드 (1080i50 | 1080p30 | 1080p25 | 720p60)")
    # 픽셀 포맷
    pixel_format: str = Field(default="yuv422", description="픽셀 포맷 (yuv422 | bgra)")
    # 사용할 SDI 임베디드 오디오 채널 번호 목록 (1~8)
    audio_channels: list[int] = Field(default=[1, 2], description="SDI 오디오 채널 번호 목록")
    # SDI 원본 오디오 샘플링레이트 (Hz)
    audio_sample_rate: int = Field(default=48000, description="SDI 원본 샘플링레이트 (Hz)")
    # SDI 원본 오디오 비트뎁스
    audio_bit_depth: int = Field(default=24, description="SDI 원본 비트뎁스")
    # 비디오 프레임 큐 최대 크기
    video_queue_size: int = Field(default=30, description="비디오 큐 최대 프레임 수")
    # 오디오 패킷 큐 최대 크기
    audio_queue_size: int = Field(default=100, description="오디오 큐 최대 패킷 수")
    # 테스트 모드 파일 설정
    test_file: TestFileConfig = Field(default_factory=TestFileConfig, description="테스트 모드 파일 설정")


# =============================================================================
# audio 섹션: 오디오 리샘플링 및 VAD 설정
# =============================================================================

class VadConfig(BaseModel):
    """
    WebRTC VAD(Voice Activity Detection) 설정입니다.

    역할:
    - 음성 구간 검출 활성화/비활성화
    - VAD 공격성 모드 설정 (0=관대, 3=엄격)
    - 음성 종료 후 패딩 시간 제어
    """
    # VAD 활성화 여부
    enabled: bool = Field(default=True, description="VAD 활성화 여부")
    # WebRTC VAD 공격성 모드 (0~3, 3이 가장 공격적으로 비음성 구간을 잘라냄)
    mode: int = Field(default=3, description="WebRTC VAD 공격성 (0~3)")
    # 음성 종료 후 추가로 전송할 패딩 시간 (밀리초)
    padding_ms: int = Field(default=300, description="음성 종료 후 패딩 (ms)")

    @field_validator("mode")
    @classmethod
    def validate_vad_mode(cls, value: int) -> int:
        """VAD 모드가 0~3 범위인지 검증합니다."""
        if not 0 <= value <= 3:
            error_message = f"VAD mode는 0~3 범위여야 합니다. 입력값: {value}"
            raise ValueError(error_message)
        return value


class AudioConfig(BaseModel):
    """
    오디오 리샘플링 출력 설정을 정의하는 모델입니다.

    역할:
    - STT 입력에 맞는 출력 포맷(16kHz/16bit/mono) 지정
    - 게인 조정 범위 제한 (-20dB ~ +20dB)
    - 청크 크기로 STT 전송 단위 결정
    - VAD 설정 포함
    """
    # STT 입력용 출력 샘플링레이트 (Hz)
    output_sample_rate: int = Field(default=16000, description="출력 샘플링레이트 (Hz)")
    # STT 입력용 출력 비트뎁스
    output_bit_depth: int = Field(default=16, description="출력 비트뎁스")
    # 출력 채널 수 (1=모노)
    output_channels: int = Field(default=1, description="출력 채널 수 (1=mono)")
    # 게인 조정값 (dB 단위, -20.0 ~ +20.0)
    gain_db: float = Field(default=0.0, description="게인 조정 (dB, -20.0 ~ +20.0)")
    # STT 전송 청크 크기 (밀리초)
    chunk_size_ms: int = Field(default=100, description="STT 전송 청크 크기 (ms)")
    # VAD 설정
    vad: VadConfig = Field(default_factory=VadConfig, description="VAD 설정")

    @field_validator("gain_db")
    @classmethod
    def validate_gain_db(cls, value: float) -> float:
        """
        게인값이 안전한 범위(-20dB ~ +20dB) 내인지 검증합니다.

        -20dB 미만은 사실상 무음이고, +20dB 초과는 클리핑이 발생하므로
        허용 범위를 제한합니다.
        """
        min_gain = -20.0
        max_gain = 20.0
        if not min_gain <= value <= max_gain:
            error_message = (
                f"gain_db는 {min_gain}~{max_gain} 범위여야 합니다. "
                f"입력값: {value}"
            )
            raise ValueError(error_message)
        return value


# =============================================================================
# stt 섹션: Clova Speech gRPC STT 설정
# =============================================================================

class STTConfig(BaseModel):
    """
    Clova Speech gRPC STT 연결 및 동작 설정입니다.

    역할:
    - gRPC 엔드포인트 및 인증 정보 관리
    - 언어/도메인/모델 선택
    - SSL, 타임아웃, 재연결 전략 설정
    - 결과 큐 크기 제한
    """
    # Clova Speech gRPC 서버 주소 (host:port)
    endpoint: str = Field(default="clovaspeech-gw.ncloud.com:50051", description="gRPC 서버 주소")
    # API 인증 키 (환경변수 SRS_STT_API_KEY 사용 권장)
    api_key: str = Field(default="", description="API 인증 키")
    # 시크릿 키 (환경변수 SRS_STT_SECRET_KEY 사용 권장)
    secret_key: str = Field(default="", description="시크릿 키")
    # STT 인식 언어
    language: str = Field(default="ko-KR", description="인식 언어 (ko-KR | en-US | ja-JP)")
    # STT 도메인 (일반/금융/의료)
    domain: str = Field(default="general", description="도메인 (general | finance | medical)")
    # STT 모델 식별자
    model: str = Field(default="general", description="모델 식별자")
    # SSL 사용 여부
    use_ssl: bool = Field(default=True, description="SSL 사용 여부")
    # gRPC 요청 타임아웃 (초)
    timeout_sec: int = Field(default=30, description="gRPC 타임아웃 (초)")
    # 최대 재연결 시도 횟수
    max_reconnect_attempts: int = Field(default=5, description="최대 재연결 시도 횟수")
    # 재연결 backoff 기본 대기 시간 (초)
    reconnect_backoff_base_sec: int = Field(default=1, description="재연결 backoff 기본값 (초)")
    # 재연결 backoff 최대 대기 시간 (초)
    reconnect_backoff_max_sec: int = Field(default=16, description="재연결 backoff 최대값 (초)")
    # STT 결과 큐 최대 크기
    result_queue_size: int = Field(default=50, description="STT 결과 큐 크기")


# =============================================================================
# subtitle 섹션: 자막 표시 및 내보내기 설정
# =============================================================================

class FontConfig(BaseModel):
    """
    자막 폰트 렌더링 설정입니다.

    역할:
    - 폰트 파일 경로, 크기, 색상 지정
    - 배경색 및 외곽선(stroke) 스타일 설정
    """
    # 폰트 파일 경로 (macOS 기본 한글 폰트)
    path: str = Field(
        default="/System/Library/Fonts/AppleSDGothicNeo.ttc",
        description="폰트 파일 경로"
    )
    # 폰트 크기 (픽셀)
    size: int = Field(default=36, description="폰트 크기")
    # 텍스트 색상 (HEX 코드)
    color: str = Field(default="#FFFFFF", description="텍스트 색상 (HEX)")
    # 텍스트 배경 색상 (HEX + 투명도)
    background_color: str = Field(default="#00000080", description="배경 색상 (HEX + 투명도)")
    # 외곽선 색상
    stroke_color: str = Field(default="#000000", description="외곽선 색상")
    # 외곽선 두께
    stroke_width: int = Field(default=2, description="외곽선 두께")


class PositionConfig(BaseModel):
    """
    자막 표시 위치 설정입니다.

    역할:
    - 화면 내 자막 위치를 정규화 좌표(0.0~1.0)로 지정
    - 앵커 포인트(center/left/right)로 정렬 방식 결정
    """
    # 가로 위치 (0.0=좌측, 0.5=중앙, 1.0=우측)
    x: float = Field(default=0.5, description="가로 위치 (0.0~1.0)")
    # 세로 위치 (0.0=상단, 0.85=하단 근처, 1.0=맨 아래)
    y: float = Field(default=0.85, description="세로 위치 (0.0~1.0)")
    # 텍스트 정렬 앵커
    anchor: str = Field(default="center", description="앵커 (center | left | right)")


class ExportConfig(BaseModel):
    """
    자막 파일 내보내기 설정입니다.

    역할:
    - SRT/VTT 형식 자막 파일 자동 저장 활성화
    - 저장 포맷 및 출력 디렉토리 지정
    """
    # 자막 파일 저장 활성화 여부
    enabled: bool = Field(default=True, description="자막 파일 저장 활성화")
    # 저장할 포맷 목록
    format: list[str] = Field(default=["srt", "vtt"], description="저장 포맷 목록")
    # 자막 파일 출력 디렉토리
    output_dir: str = Field(default="output/subtitles", description="자막 파일 출력 디렉토리")


class SubtitleConfig(BaseModel):
    """
    자막 표시 동작 및 스타일 설정을 정의하는 모델입니다.

    역할:
    - partial/final 결과 표시 제어
    - 동기화 오프셋 및 표시 지속 시간 설정
    - 폰트, 위치, 내보내기 하위 설정 포함
    """
    # partial(중간) 결과 화면 표시 여부
    show_partial: bool = Field(default=True, description="partial 결과 표시 여부")
    # partial 결과 갱신 주기 (밀리초)
    partial_update_interval_ms: int = Field(default=200, description="partial 갱신 주기 (ms)")
    # 자막 표시 시점 오프셋 (양수=지연, 음수=앞당김)
    sync_offset_ms: int = Field(default=0, description="동기화 오프셋 (ms)")
    # final 자막 표시 유지 시간 (밀리초)
    display_duration_ms: int = Field(default=3000, description="자막 표시 유지 시간 (ms)")
    # 자막 히스토리 버퍼 최대 크기
    history_size: int = Field(default=100, description="히스토리 버퍼 크기")
    # 폰트 설정
    font: FontConfig = Field(default_factory=FontConfig, description="폰트 설정")
    # 위치 설정
    position: PositionConfig = Field(default_factory=PositionConfig, description="위치 설정")
    # 내보내기 설정
    export: ExportConfig = Field(default_factory=ExportConfig, description="내보내기 설정")


# =============================================================================
# accuracy 섹션: WER/CER 정확도 측정 설정
# =============================================================================

class AccuracyConfig(BaseModel):
    """
    WER/CER 정확도 측정 설정입니다.

    역할:
    - 정확도 측정 기능 활성화/비활성화
    - 레퍼런스 텍스트 소스(파일/실시간) 지정
    - 리포트 출력 주기 및 저장 경로 설정
    """
    # 정확도 측정 활성화 여부
    enabled: bool = Field(default=False, description="정확도 측정 활성화")
    # 레퍼런스 텍스트 소스 유형
    reference_source: str = Field(default="file", description="레퍼런스 소스 (file | realtime)")
    # 레퍼런스 텍스트 파일 경로 (reference_source=file일 때 사용)
    reference_file: str = Field(default="", description="레퍼런스 파일 경로")
    # 중간 리포트 출력 주기 (초)
    report_interval_sec: int = Field(default=60, description="리포트 출력 주기 (초)")
    # 리포트 저장 디렉토리
    output_dir: str = Field(default="output/reports", description="리포트 출력 디렉토리")


# =============================================================================
# metrics 섹션: 메트릭 수집 설정
# =============================================================================

class PrometheusConfig(BaseModel):
    """
    Prometheus 메트릭 수집 설정입니다 (Phase 3에서 활성화 예정).

    역할:
    - Prometheus 메트릭 엔드포인트 활성화
    - 수집 서버 포트 지정
    """
    # Prometheus 활성화 여부
    enabled: bool = Field(default=False, description="Prometheus 활성화 (Phase 3)")
    # Prometheus 메트릭 서버 포트
    port: int = Field(default=9090, description="Prometheus 포트")


class MetricsConfig(BaseModel):
    """
    지연시간 및 성능 메트릭 수집 설정입니다.

    역할:
    - 지연시간 통계 계산 윈도우 크기 설정
    - 메트릭 리포트 저장 경로 지정
    - Prometheus 연동 설정 포함
    """
    # 지연시간 통계 슬라이딩 윈도우 크기 (초)
    latency_window_sec: int = Field(default=60, description="지연시간 통계 윈도우 (초)")
    # 메트릭 리포트 저장 디렉토리
    metrics_output_dir: str = Field(default="output/reports", description="메트릭 출력 디렉토리")
    # Prometheus 설정
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig, description="Prometheus 설정")


# =============================================================================
# dashboard 섹션: 대시보드 표시 설정
# =============================================================================

class WebDashboardConfig(BaseModel):
    """
    웹 대시보드 서버 설정입니다 (Phase 3에서 구현 예정).

    역할:
    - 웹 대시보드 바인드 주소 및 포트 지정
    """
    # 웹 서버 바인드 호스트 주소
    host: str = Field(default="0.0.0.0", description="웹 서버 호스트")
    # 웹 서버 포트
    port: int = Field(default=8080, description="웹 서버 포트")


class DashboardConfig(BaseModel):
    """
    대시보드 유형 및 갱신 주기 설정입니다.

    역할:
    - TUI/Web/None 중 대시보드 유형 선택
    - 갱신 주기로 리소스 사용량 제어
    - 웹 대시보드 하위 설정 포함
    """
    # 대시보드 유형 (tui=터미널, web=브라우저, none=비활성)
    type: str = Field(default="tui", description="대시보드 유형 (tui | web | none)")
    # 대시보드 갱신 주기 (밀리초)
    refresh_interval_ms: int = Field(default=500, description="갱신 주기 (ms)")
    # 오디오 레벨 미터 갱신 주기 (밀리초)
    audio_meter_interval_ms: int = Field(default=100, description="오디오 미터 갱신 주기 (ms)")
    # 웹 대시보드 설정
    web: WebDashboardConfig = Field(default_factory=WebDashboardConfig, description="웹 대시보드 설정")


# =============================================================================
# alerts 섹션: 알림 임계값 설정
# =============================================================================

class AlertsConfig(BaseModel):
    """
    파이프라인 이상 상태 감지를 위한 알림 임계값 설정입니다.

    역할:
    - STT 오류, 프레임 드롭, 지연시간, 무음 감지 임계값 정의
    - 임계값 초과 시 대시보드 알림 트리거에 사용
    """
    # 연속 STT 오류 횟수 임계값 (이 값 초과 시 알림)
    stt_error_threshold: int = Field(default=3, description="연속 STT 오류 알림 임계값")
    # 프레임 드롭율 임계값 (비율, 0.01 = 1%)
    frame_drop_rate_threshold: float = Field(default=0.01, description="프레임 드롭율 임계값")
    # End-to-end 지연시간 알림 임계값 (밀리초)
    e2e_latency_alert_ms: int = Field(default=3000, description="E2E 지연 알림 임계값 (ms)")
    # 무음 지속 시간 알림 임계값 (초)
    audio_silence_sec: int = Field(default=5, description="무음 지속 알림 임계값 (초)")


# =============================================================================
# 최상위 AppConfig: 모든 섹션을 통합하는 루트 모델
# =============================================================================

class AppConfig(BaseModel):
    """
    애플리케이션 전체 설정을 통합하는 최상위 모델입니다.

    역할:
    - config.yaml의 모든 섹션을 하나의 타입 안전한 객체로 통합
    - Pydantic v2 유효성 검증을 통해 설정 무결성 보장
    - 각 섹션이 누락된 경우 기본값으로 자동 생성

    사용 예시:
        >>> import yaml
        >>> with open("config.yaml") as f:
        ...     raw = yaml.safe_load(f)
        >>> config = AppConfig(**raw)
        >>> print(config.system.mode)
        'live'
        >>> print(config.audio.gain_db)
        0.0
    """
    # 시스템 전역 설정
    system: SystemConfig = Field(default_factory=SystemConfig, description="시스템 설정")
    # SDI 캡처 설정
    capture: CaptureConfig = Field(default_factory=CaptureConfig, description="캡처 설정")
    # 오디오 리샘플링 설정
    audio: AudioConfig = Field(default_factory=AudioConfig, description="오디오 설정")
    # Clova Speech STT 설정
    stt: STTConfig = Field(default_factory=STTConfig, description="STT 설정")
    # 자막 표시 및 내보내기 설정
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig, description="자막 설정")
    # WER/CER 정확도 측정 설정
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig, description="정확도 설정")
    # 메트릭 수집 설정
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="메트릭 설정")
    # 대시보드 설정
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig, description="대시보드 설정")
    # 알림 임계값 설정
    alerts: AlertsConfig = Field(default_factory=AlertsConfig, description="알림 설정")
