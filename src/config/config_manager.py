"""
SDI-RealtimeSubtitle 설정 관리 모듈입니다.

역할:
- YAML 설정 파일을 로드하고 Pydantic 스키마로 유효성 검증
- 환경변수 오버라이드 지원 (접두사: SRS_)
- dot-notation 기반 설정값 조회 (예: "stt.api_key")
- watchdog 기반 파일 변경 감지 및 핫스왑 (Phase 3 구현)
- 설정 변경 시 구독자(콜백) 통보

사용 예시:
    >>> manager = ConfigManager()
    >>> config = manager.load("config.yaml")
    >>> api_key = manager.get("stt.api_key")
    >>> manager.subscribe(lambda old, new: print("설정 변경됨"))
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import ValidationError

from src.config.schema import AppConfig

# 모듈 로거 설정
logger = logging.getLogger(__name__)

# 환경변수 오버라이드 접두사
ENV_PREFIX = "SRS_"

# 설정 변경 콜백 타입: (이전 설정, 새 설정) -> None
ConfigChangeCallback = Callable[[AppConfig, AppConfig], None]


class ConfigLoadError(Exception):
    """설정 파일 로드 중 발생하는 에러의 기본 클래스입니다."""
    pass


class ConfigValidationError(ConfigLoadError):
    """설정 스키마 검증 실패 시 발생하는 에러입니다."""
    pass


class ConfigFileNotFoundError(ConfigLoadError):
    """설정 파일을 찾을 수 없을 때 발생하는 에러입니다."""
    pass


class ConfigManager:
    """
    YAML 설정 파일을 로드하고 관리하는 매니저 클래스입니다.

    역할:
    - YAML 파일 파싱 및 Pydantic 유효성 검증
    - 환경변수 오버라이드 (SRS_ 접두사)
    - dot-notation 설정값 조회
    - 파일 변경 감지(watchdog) 및 구독자 통보
    - 검증 실패 시 이전 설정 유지 (안전한 롤백)

    사용 예시:
        >>> manager = ConfigManager()
        >>> config = manager.load("config.yaml")
        >>> print(manager.get("stt.endpoint"))
        'clovaspeech-gw.ncloud.com:50051'
    """

    def __init__(self) -> None:
        """ConfigManager를 초기화합니다."""
        # 현재 활성 설정 객체 (로드 전에는 None)
        self._config: Optional[AppConfig] = None
        # 설정 파일 경로 (로드 시 설정됨)
        self._config_filepath: Optional[Path] = None
        # 설정 변경 시 호출할 콜백 목록
        self._subscribers: list[ConfigChangeCallback] = []
        # 설정 접근 시 스레드 안전성을 보장하기 위한 락
        self._lock: threading.RLock = threading.RLock()
        # watchdog Observer 인스턴스 (watch() 호출 시 생성)
        self._observer: Optional[Any] = None

        logger.debug("ConfigManager 인스턴스 생성 완료")

    @property
    def config(self) -> Optional[AppConfig]:
        """현재 활성 설정 객체를 반환합니다."""
        with self._lock:
            return self._config

    def load(self, filepath: str | Path) -> AppConfig:
        """
        YAML 설정 파일을 로드하고 Pydantic 스키마로 검증합니다.

        처리 순서:
        1. 파일 존재 여부 확인
        2. YAML 파싱
        3. 환경변수 오버라이드 적용
        4. Pydantic 스키마 검증
        5. 검증 통과 시 활성 설정으로 교체

        파라미터:
            filepath (str | Path): YAML 설정 파일 경로

        반환값:
            AppConfig: 검증 완료된 설정 객체

        에러:
            ConfigFileNotFoundError: 파일이 존재하지 않을 때
            ConfigValidationError: 스키마 검증 실패 시
            ConfigLoadError: YAML 파싱 실패 등 기타 에러
        """
        filepath = Path(filepath)
        logger.info(f"설정 파일 로드 시작: {filepath}")

        # 1단계: 파일 존재 여부 확인
        if not filepath.exists():
            error_message = f"설정 파일을 찾을 수 없습니다: {filepath}"
            logger.error(error_message)
            raise ConfigFileNotFoundError(error_message)

        try:
            # 2단계: YAML 파일 파싱
            raw_config = self._parse_yaml_file(filepath)
            logger.debug(f"YAML 파싱 완료: {len(raw_config)} 개 최상위 키")

            # 3단계: 환경변수 오버라이드 적용
            raw_config = self._apply_env_overrides(raw_config)

            # 4단계: Pydantic 스키마 검증
            validated_config = self._validate_config(raw_config)
            logger.info(f"설정 검증 완료: mode={validated_config.system.mode}")

            # 5단계: 활성 설정으로 교체 (스레드 안전)
            with self._lock:
                self._config = validated_config
                self._config_filepath = filepath

            logger.info(
                f"설정 로드 성공: "
                f"mode={validated_config.system.mode}, "
                f"log_level={validated_config.system.log_level}, "
                f"stt_endpoint={validated_config.stt.endpoint}"
            )
            return validated_config

        except ConfigLoadError:
            # ConfigLoadError 하위 클래스는 그대로 전파
            raise

        except yaml.YAMLError as yaml_error:
            error_message = f"YAML 파싱 실패: {yaml_error}"
            logger.error(error_message, exc_info=True)
            raise ConfigLoadError(error_message) from yaml_error

        except Exception as unexpected_error:
            error_message = f"설정 로드 중 예상치 못한 에러: {unexpected_error}"
            logger.error(error_message, exc_info=True)
            raise ConfigLoadError(error_message) from unexpected_error

    def get(self, key: str, default: Any = None) -> Any:
        """
        dot-notation으로 설정값을 조회합니다.

        중첩된 설정값에 접근할 때 점(.)으로 구분된 키를 사용합니다.
        예: "stt.api_key" -> config.stt.api_key

        파라미터:
            key (str): dot-notation 설정 키 (예: "stt.api_key", "audio.gain_db")
            default (Any): 키가 존재하지 않을 때 반환할 기본값

        반환값:
            Any: 설정값 또는 기본값

        에러:
            RuntimeError: 설정이 로드되지 않은 상태에서 호출 시
        """
        with self._lock:
            # 설정이 로드되었는지 확인
            if self._config is None:
                error_message = "설정이 아직 로드되지 않았습니다. load()를 먼저 호출하세요."
                logger.error(error_message)
                raise RuntimeError(error_message)

            # dot-notation 키를 분할하여 순차적으로 탐색
            key_parts = key.split(".")
            current_value: Any = self._config

            for part in key_parts:
                try:
                    # Pydantic 모델인 경우 getattr로 접근
                    if hasattr(current_value, part):
                        current_value = getattr(current_value, part)
                    # dict인 경우 키로 접근
                    elif isinstance(current_value, dict):
                        current_value = current_value[part]
                    else:
                        # 해당 키를 찾을 수 없는 경우
                        logger.debug(f"설정 키 '{key}'에서 '{part}' 부분을 찾을 수 없음, 기본값 반환")
                        return default
                except (AttributeError, KeyError, TypeError):
                    logger.debug(f"설정 키 '{key}' 조회 실패, 기본값 반환")
                    return default

            return current_value

    def subscribe(self, callback: ConfigChangeCallback) -> None:
        """
        설정 변경 시 호출될 콜백 함수를 등록합니다.

        등록된 콜백은 설정이 핫스왑(파일 변경 감지 후 리로드)될 때
        이전 설정과 새 설정을 인자로 받아 호출됩니다.

        파라미터:
            callback (ConfigChangeCallback): (이전_설정, 새_설정) -> None 형태의 콜백
        """
        self._subscribers.append(callback)
        logger.info(f"설정 변경 구독자 등록 완료 (총 {len(self._subscribers)}명)")

    def unsubscribe(self, callback: ConfigChangeCallback) -> None:
        """
        등록된 설정 변경 콜백을 제거합니다.

        파라미터:
            callback (ConfigChangeCallback): 제거할 콜백 함수
        """
        try:
            self._subscribers.remove(callback)
            logger.info(f"설정 변경 구독자 제거 완료 (남은 구독자: {len(self._subscribers)}명)")
        except ValueError:
            logger.warning("제거할 구독자를 찾을 수 없습니다")

    def watch(self, filepath: str | Path | None = None) -> None:
        """
        watchdog을 사용하여 설정 파일 변경을 감시합니다.

        파일이 수정되면 자동으로 리로드하고, 검증 통과 시
        등록된 구독자들에게 변경 사항을 통보합니다.

        현재는 Phase 3용 stub 구현이며, watchdog 의존성이 설치된 경우에만
        실제 감시를 시작합니다.

        파라미터:
            filepath (str | Path | None): 감시할 파일 경로.
                None이면 마지막으로 로드한 파일 경로를 사용합니다.
        """
        # 감시할 파일 경로 결정
        watch_path = Path(filepath) if filepath else self._config_filepath

        if watch_path is None:
            logger.warning("감시할 파일 경로가 지정되지 않았습니다. load()를 먼저 호출하세요.")
            return

        logger.info(f"설정 파일 감시 시작 (stub): {watch_path}")

        try:
            # watchdog 라이브러리가 설치되어 있으면 실제 감시 시작
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            # 파일 변경 이벤트 핸들러 정의
            config_manager_ref = self
            target_filename = watch_path.name

            class _ConfigFileHandler(FileSystemEventHandler):
                """설정 파일 변경을 감지하는 이벤트 핸들러입니다."""

                def on_modified(self, event: Any) -> None:
                    """파일 수정 이벤트를 처리합니다."""
                    # 디렉토리 이벤트는 무시
                    if event.is_directory:
                        return
                    # 대상 파일만 처리
                    if Path(event.src_path).name == target_filename:
                        logger.info(f"설정 파일 변경 감지: {event.src_path}")
                        config_manager_ref._on_file_changed(event)

            # Observer 생성 및 시작
            observer = Observer()
            observer.schedule(
                _ConfigFileHandler(),
                path=str(watch_path.parent),
                recursive=False,
            )
            observer.daemon = True
            observer.start()
            self._observer = observer

            logger.info(f"설정 파일 감시 활성화 완료: {watch_path}")

        except ImportError:
            logger.warning(
                "watchdog 라이브러리가 설치되지 않아 파일 감시를 시작할 수 없습니다. "
                "pip install watchdog 으로 설치하세요."
            )

    def stop_watch(self) -> None:
        """설정 파일 감시를 중지합니다."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("설정 파일 감시 중지 완료")

    def validate_schema(self, raw_config: dict) -> bool:
        """
        딕셔너리 데이터가 AppConfig 스키마를 만족하는지 검증합니다.

        파라미터:
            raw_config (dict): 검증할 설정 딕셔너리

        반환값:
            bool: 검증 통과 시 True, 실패 시 False
        """
        try:
            AppConfig(**raw_config)
            logger.debug("스키마 검증 통과")
            return True
        except ValidationError as validation_error:
            logger.warning(f"스키마 검증 실패: {validation_error}")
            return False

    # =========================================================================
    # 내부 메서드 (private)
    # =========================================================================

    def _parse_yaml_file(self, filepath: Path) -> dict:
        """
        YAML 파일을 읽어서 딕셔너리로 파싱합니다.

        파라미터:
            filepath (Path): YAML 파일 경로

        반환값:
            dict: 파싱된 설정 딕셔너리

        에러:
            ConfigLoadError: 파일 읽기 또는 파싱 실패 시
        """
        try:
            with open(filepath, "r", encoding="utf-8") as config_file:
                raw_data = yaml.safe_load(config_file)

            # YAML 파일이 비어있거나 파싱 결과가 None인 경우 빈 딕셔너리 반환
            if raw_data is None:
                logger.warning(f"설정 파일이 비어있습니다: {filepath}")
                return {}

            # 파싱 결과가 딕셔너리가 아닌 경우 에러
            if not isinstance(raw_data, dict):
                error_message = f"설정 파일의 최상위 구조가 딕셔너리가 아닙니다: {type(raw_data)}"
                raise ConfigLoadError(error_message)

            return raw_data

        except yaml.YAMLError as yaml_error:
            error_message = f"YAML 파싱 에러: {yaml_error}"
            logger.error(error_message, exc_info=True)
            raise ConfigLoadError(error_message) from yaml_error

        except OSError as file_error:
            error_message = f"파일 읽기 에러: {file_error}"
            logger.error(error_message, exc_info=True)
            raise ConfigLoadError(error_message) from file_error

    def _apply_env_overrides(self, raw_config: dict) -> dict:
        """
        SRS_ 접두사 환경변수로 설정값을 오버라이드합니다.

        환경변수 매핑 규칙:
        - 접두사: SRS_
        - 구분자: _ (언더스코어)
        - 대소문자: 환경변수는 대문자, 설정 키는 소문자로 변환
        - 예: SRS_STT_API_KEY -> stt.api_key
        - 예: SRS_SYSTEM_MODE -> system.mode
        - 예: SRS_AUDIO_GAIN_DB -> audio.gain_db

        파라미터:
            raw_config (dict): 환경변수 적용 전 설정 딕셔너리

        반환값:
            dict: 환경변수가 적용된 설정 딕셔너리
        """
        # 오버라이드 적용 횟수 추적
        override_count = 0

        # SRS_ 접두사를 가진 모든 환경변수 수집
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(ENV_PREFIX):
                continue

            # 접두사 제거 후 소문자로 변환하여 설정 경로 생성
            # 예: SRS_STT_API_KEY -> stt_api_key
            config_path = env_key[len(ENV_PREFIX):].lower()

            # 첫 번째 언더스코어를 섹션 구분자로 사용
            # 예: stt_api_key -> section="stt", remaining="api_key"
            path_parts = config_path.split("_", 1)

            if len(path_parts) < 2:
                # 섹션만 있고 키가 없는 경우 무시
                logger.debug(f"환경변수 '{env_key}' 무시 (키 경로 부족)")
                continue

            section_name = path_parts[0]
            field_name = path_parts[1]

            # 해당 섹션이 설정에 존재하는지 확인
            if section_name not in raw_config:
                # 섹션이 없으면 새로 생성
                raw_config[section_name] = {}

            # 설정값 타입 자동 변환
            converted_value = self._convert_env_value(env_value)

            # 중첩 키 처리 (예: audio_vad_mode -> audio.vad.mode)
            # 2단계 중첩만 지원 (section.subsection.field)
            if isinstance(raw_config.get(section_name), dict):
                sub_parts = field_name.split("_", 1)
                # 하위 섹션이 이미 존재하는지 확인
                if (
                    len(sub_parts) == 2
                    and sub_parts[0] in raw_config[section_name]
                    and isinstance(raw_config[section_name][sub_parts[0]], dict)
                ):
                    # 하위 섹션의 필드를 오버라이드
                    raw_config[section_name][sub_parts[0]][sub_parts[1]] = converted_value
                    logger.info(
                        f"환경변수 오버라이드: {env_key} -> "
                        f"{section_name}.{sub_parts[0]}.{sub_parts[1]} = "
                        f"{'***' if 'key' in field_name.lower() else converted_value}"
                    )
                else:
                    # 1단계 중첩 필드를 오버라이드
                    raw_config[section_name][field_name] = converted_value
                    logger.info(
                        f"환경변수 오버라이드: {env_key} -> "
                        f"{section_name}.{field_name} = "
                        f"{'***' if 'key' in field_name.lower() else converted_value}"
                    )
            override_count += 1

        if override_count > 0:
            logger.info(f"환경변수 오버라이드 적용 완료: {override_count}건")

        return raw_config

    def _convert_env_value(self, value: str) -> Any:
        """
        환경변수 문자열 값을 적절한 Python 타입으로 변환합니다.

        변환 규칙:
        - "true"/"false" (대소문자 무관) -> bool
        - 정수 형식 문자열 -> int
        - 부동소수점 형식 문자열 -> float
        - 그 외 -> str (원본 유지)

        파라미터:
            value (str): 환경변수 값

        반환값:
            Any: 변환된 Python 값
        """
        # 불리언 변환
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # 정수 변환 시도
        try:
            return int(value)
        except ValueError:
            pass

        # 부동소수점 변환 시도
        try:
            return float(value)
        except ValueError:
            pass

        # 변환 불가 시 문자열 그대로 반환
        return value

    def _validate_config(self, raw_config: dict) -> AppConfig:
        """
        딕셔너리를 Pydantic AppConfig 모델로 검증하고 변환합니다.

        파라미터:
            raw_config (dict): 검증할 설정 딕셔너리

        반환값:
            AppConfig: 검증 완료된 설정 객체

        에러:
            ConfigValidationError: Pydantic 검증 실패 시
        """
        try:
            validated = AppConfig(**raw_config)
            return validated

        except ValidationError as validation_error:
            # 검증 에러 상세 내용을 로그에 기록
            error_details = validation_error.errors()
            for error_detail in error_details:
                field_path = " -> ".join(str(loc) for loc in error_detail["loc"])
                logger.error(
                    f"설정 검증 실패 - 필드: {field_path}, "
                    f"에러: {error_detail['msg']}, "
                    f"입력값: {error_detail.get('input', 'N/A')}"
                )

            error_message = f"설정 스키마 검증 실패: {len(error_details)}개 에러 발생"
            raise ConfigValidationError(error_message) from validation_error

    def _on_file_changed(self, event: Any) -> None:
        """
        설정 파일 변경 감지 시 호출되는 핸들러입니다.

        처리 순서:
        1. 변경된 파일을 다시 파싱
        2. Pydantic 스키마 검증
        3. 검증 통과 시 활성 설정 교체
        4. 등록된 구독자들에게 변경 통보
        5. 검증 실패 시 이전 설정 유지 (안전한 롤백)

        파라미터:
            event: watchdog 파일 시스템 이벤트 객체
        """
        if self._config_filepath is None:
            logger.warning("설정 파일 경로가 설정되지 않아 리로드를 건너뜁니다")
            return

        logger.info(f"설정 파일 변경 감지, 리로드 시작: {self._config_filepath}")

        try:
            # 1단계: YAML 파싱
            raw_config = self._parse_yaml_file(self._config_filepath)

            # 2단계: 환경변수 오버라이드 적용
            raw_config = self._apply_env_overrides(raw_config)

            # 3단계: 스키마 검증
            new_config = self._validate_config(raw_config)

            # 4단계: 이전 설정 백업 및 새 설정 적용 (스레드 안전)
            with self._lock:
                previous_config = self._config
                self._config = new_config

            logger.info("설정 핫스왑 성공: 새 설정이 적용되었습니다")

            # 5단계: 구독자들에게 변경 통보
            if previous_config is not None:
                self._notify_subscribers(previous_config, new_config)

        except (ConfigLoadError, ConfigValidationError) as load_error:
            # 검증 실패 시 이전 설정 유지 (롤백)
            logger.error(
                f"설정 핫스왑 실패, 이전 설정을 유지합니다: {load_error}"
            )

        except Exception as unexpected_error:
            # 예상치 못한 에러에도 이전 설정 유지
            logger.error(
                f"설정 리로드 중 예상치 못한 에러, 이전 설정 유지: {unexpected_error}",
                exc_info=True,
            )

    def _notify_subscribers(
        self,
        previous_config: AppConfig,
        new_config: AppConfig,
    ) -> None:
        """
        등록된 모든 구독자에게 설정 변경을 통보합니다.

        개별 구독자의 콜백 실행 중 에러가 발생해도
        다른 구독자의 통보는 계속 진행합니다.

        파라미터:
            previous_config (AppConfig): 변경 전 설정
            new_config (AppConfig): 변경 후 설정
        """
        subscriber_count = len(self._subscribers)
        logger.info(f"설정 변경 통보 시작: {subscriber_count}명의 구독자")

        for subscriber_index, callback in enumerate(self._subscribers):
            try:
                callback(previous_config, new_config)
                logger.debug(f"구독자 {subscriber_index + 1}/{subscriber_count} 통보 완료")
            except Exception as callback_error:
                # 개별 구독자 에러가 다른 구독자에 영향을 주지 않도록 격리
                logger.error(
                    f"구독자 {subscriber_index + 1} 콜백 실행 중 에러: {callback_error}",
                    exc_info=True,
                )

        logger.info("설정 변경 통보 완료")
