"""
구조화 JSON 로깅 모듈입니다.

역할:
- python-json-logger를 사용한 JSON 포맷 로그 출력
- RotatingFileHandler로 로그 파일 자동 순환 (10MB, 5개 보존)
- session_id, module, level 등 공통 필드 자동 추가
- 로그 레벨 및 포맷(json/text)을 설정에서 제어

사용 예시:
    >>> setup_logging(config)
    >>> logger = StructuredLogger.get("my_module")
    >>> logger.info("처리 시작", extra={"packet_id": 1})
"""

from __future__ import annotations

import logging
import logging.handlers
import uuid
from pathlib import Path
from typing import Optional

from pythonjsonlogger import jsonlogger

from src.config.schema import AppConfig

_SESSION_ID: str = ""


def setup_logging(config: AppConfig, session_id: Optional[str] = None) -> None:
    """
    애플리케이션 전체 로깅 설정을 초기화합니다.

    파라미터:
        config: AppConfig 인스턴스
        session_id: 세션 식별자. None이면 config.system.session_id 또는 UUID 사용
    """
    global _SESSION_ID

    _SESSION_ID = (
        session_id
        or config.system.session_id
        or str(uuid.uuid4())
    )

    log_level = getattr(logging, config.system.log_level, logging.INFO)
    log_format = config.system.log_format
    log_dir = Path(config.system.log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 핸들러 생성
    handlers: list[logging.Handler] = []

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # 파일 핸들러 (RotatingFileHandler: 10MB, 5개 보존)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filepath = log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_filepath,
            maxBytes=10 * 1024 * 1024,   # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    except OSError as exc:
        logging.warning(f"로그 파일 핸들러 생성 실패: {exc}")

    # 포맷터 적용
    for handler in handlers:
        if log_format == "json":
            formatter = _JsonFormatter(session_id=_SESSION_ID)
        else:
            formatter = _TextFormatter(session_id=_SESSION_ID)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    logging.getLogger(__name__).info(
        f"로깅 초기화: level={config.system.log_level}, "
        f"format={log_format}, session={_SESSION_ID}"
    )


class _JsonFormatter(jsonlogger.JsonFormatter):
    """
    session_id, module 필드를 자동 추가하는 JSON 포맷터입니다.
    """

    def __init__(self, session_id: str = "") -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self._session_id = session_id

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["session_id"] = self._session_id
        log_record["module"] = record.name
        log_record["level"] = record.levelname


class _TextFormatter(logging.Formatter):
    """
    session_id를 접두어로 포함하는 텍스트 포맷터입니다.
    """

    def __init__(self, session_id: str = "") -> None:
        super().__init__(
            fmt=f"%(asctime)s [{session_id[:8] if session_id else 'no-sid'}] "
                f"%(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class StructuredLogger:
    """
    모듈별 구조화 로거를 반환하는 팩토리 클래스입니다.

    get() 메서드로 표준 logging.Logger를 래핑하지 않고
    표준 Logger를 직접 반환하여 기존 logging API와 완전히 호환됩니다.
    """

    @staticmethod
    def get(name: str) -> logging.Logger:
        """
        지정된 이름의 로거를 반환합니다.

        파라미터:
            name: 모듈/컴포넌트 이름 (일반적으로 __name__ 사용)

        반환값:
            logging.Logger: 표준 로거 인스턴스
        """
        return logging.getLogger(name)

    @staticmethod
    def get_session_id() -> str:
        """현재 세션 ID를 반환합니다."""
        return _SESSION_ID
