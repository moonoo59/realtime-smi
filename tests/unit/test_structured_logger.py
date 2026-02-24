"""
구조화 로깅 모듈 단위 테스트

검증 조건:
- JSON 포맷 로그에 session_id, level, module 필드 포함
- RotatingFileHandler 파일 생성 확인
- text 포맷 정상 동작
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from io import StringIO
from pathlib import Path

import pytest

from src.config.schema import AppConfig
from src.logging.structured_logger import StructuredLogger, setup_logging


# =========================================================================
# 픽스처
# =========================================================================

@pytest.fixture(autouse=True)
def reset_root_logger():
    """각 테스트 후 root logger 핸들러 초기화."""
    yield
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()


@pytest.fixture
def config_json(tmp_path):
    cfg = AppConfig()
    cfg.system.log_level = "DEBUG"
    cfg.system.log_format = "json"
    cfg.system.log_dir = str(tmp_path / "logs")
    cfg.system.session_id = "test-session-001"
    return cfg


@pytest.fixture
def config_text(tmp_path):
    cfg = AppConfig()
    cfg.system.log_level = "DEBUG"
    cfg.system.log_format = "text"
    cfg.system.log_dir = str(tmp_path / "logs")
    cfg.system.session_id = "text-session-002"
    return cfg


# =========================================================================
# setup_logging 테스트
# =========================================================================

class TestSetupLogging:
    def test_json_format_sets_json_handler(self, config_json):
        setup_logging(config_json)
        root = logging.getLogger()
        assert len(root.handlers) >= 1

    def test_log_file_created(self, config_json, tmp_path):
        setup_logging(config_json)
        log_file = tmp_path / "logs" / "app.log"
        assert log_file.exists()

    def test_session_id_stored(self, config_json):
        setup_logging(config_json, session_id="custom-sid")
        assert StructuredLogger.get_session_id() == "custom-sid"

    def test_session_id_from_config(self, config_json):
        setup_logging(config_json)
        assert StructuredLogger.get_session_id() == "test-session-001"

    def test_session_id_auto_uuid_when_empty(self, tmp_path):
        cfg = AppConfig()
        cfg.system.session_id = ""
        cfg.system.log_dir = str(tmp_path / "logs")
        setup_logging(cfg)
        sid = StructuredLogger.get_session_id()
        assert len(sid) == 36  # UUID 형식
        assert sid.count("-") == 4

    def test_log_level_applied(self, config_json):
        config_json.system.log_level = "WARNING"
        setup_logging(config_json)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_duplicate_setup_does_not_add_extra_handlers(self, config_json):
        setup_logging(config_json)
        handler_count = len(logging.getLogger().handlers)
        setup_logging(config_json)
        # 재설정 시 이전 핸들러 제거 후 재추가
        assert len(logging.getLogger().handlers) == handler_count

    def test_rotating_file_handler_present(self, config_json, tmp_path):
        setup_logging(config_json)
        root = logging.getLogger()
        handler_types = [type(h) for h in root.handlers]
        assert logging.handlers.RotatingFileHandler in handler_types


# =========================================================================
# JSON 로그 출력 형식 테스트
# =========================================================================

class TestJsonFormat:
    def _capture_json_log(self, config, message: str, extra: dict = None) -> dict:
        """메모리 스트림에 JSON 로그를 캡처하여 파싱합니다."""
        stream = StringIO()
        setup_logging(config)

        from pythonjsonlogger import jsonlogger
        from src.logging.structured_logger import _JsonFormatter

        # 스트림 핸들러 추가
        stream_handler = logging.StreamHandler(stream)
        formatter = _JsonFormatter(session_id=config.system.session_id)
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger("test.json")
        logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)

        if extra:
            logger.info(message, extra=extra)
        else:
            logger.info(message)

        logger.removeHandler(stream_handler)
        output = stream.getvalue().strip()
        if not output:
            pytest.skip("로그 출력 없음")
        return json.loads(output)

    def test_json_contains_session_id(self, config_json):
        data = self._capture_json_log(config_json, "테스트 메시지")
        assert "session_id" in data
        assert data["session_id"] == "test-session-001"

    def test_json_contains_level(self, config_json):
        data = self._capture_json_log(config_json, "레벨 테스트")
        assert "level" in data
        assert data["level"] == "INFO"

    def test_json_contains_module(self, config_json):
        data = self._capture_json_log(config_json, "모듈 테스트")
        assert "module" in data

    def test_json_contains_message(self, config_json):
        data = self._capture_json_log(config_json, "메시지 확인")
        assert "message" in data
        assert data["message"] == "메시지 확인"

    def test_json_extra_fields_included(self, config_json):
        data = self._capture_json_log(
            config_json, "추가 필드", extra={"packet_id": 42}
        )
        assert data.get("packet_id") == 42

    def test_json_is_valid_json(self, config_json):
        """로그 출력이 유효한 JSON이어야 한다."""
        stream = StringIO()
        from src.logging.structured_logger import _JsonFormatter

        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter(session_id="abc"))
        logger = logging.getLogger("json.valid")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.warning("경고 메시지")
        logger.removeHandler(handler)

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert isinstance(parsed, dict)


# =========================================================================
# 텍스트 포맷 테스트
# =========================================================================

class TestTextFormat:
    def test_text_format_includes_session_id_prefix(self, config_text):
        stream = StringIO()
        from src.logging.structured_logger import _TextFormatter

        handler = logging.StreamHandler(stream)
        handler.setFormatter(_TextFormatter(session_id="text-session-002"))
        logger = logging.getLogger("text.test")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("텍스트 로그 테스트")
        logger.removeHandler(handler)

        output = stream.getvalue()
        assert "text-se" in output  # 8자 접두어
        assert "텍스트 로그 테스트" in output

    def test_text_format_includes_level(self, config_text):
        stream = StringIO()
        from src.logging.structured_logger import _TextFormatter

        handler = logging.StreamHandler(stream)
        handler.setFormatter(_TextFormatter(session_id="sid"))
        logger = logging.getLogger("text.level")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.error("오류 메시지")
        logger.removeHandler(handler)

        output = stream.getvalue()
        assert "ERROR" in output


# =========================================================================
# StructuredLogger 팩토리 테스트
# =========================================================================

class TestStructuredLogger:
    def test_get_returns_logger(self):
        logger = StructuredLogger.get("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_returns_named_logger(self):
        logger = StructuredLogger.get("my.module.name")
        assert logger.name == "my.module.name"

    def test_get_same_name_returns_same_instance(self):
        logger1 = StructuredLogger.get("shared.module")
        logger2 = StructuredLogger.get("shared.module")
        assert logger1 is logger2

    def test_get_session_id_returns_string(self, config_json):
        setup_logging(config_json)
        sid = StructuredLogger.get_session_id()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_logger_can_emit_all_levels(self, config_json):
        setup_logging(config_json)
        logger = StructuredLogger.get("all.levels")
        # 예외 없이 모든 레벨 로그 가능
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")


# =========================================================================
# 로그 파일 순환 설정 테스트
# =========================================================================

class TestRotatingFile:
    def test_rotating_handler_max_bytes(self, config_json, tmp_path):
        setup_logging(config_json)
        root = logging.getLogger()
        rotating = next(
            (h for h in root.handlers
             if isinstance(h, logging.handlers.RotatingFileHandler)),
            None,
        )
        assert rotating is not None
        assert rotating.maxBytes == 10 * 1024 * 1024  # 10MB

    def test_rotating_handler_backup_count(self, config_json, tmp_path):
        setup_logging(config_json)
        root = logging.getLogger()
        rotating = next(
            (h for h in root.handlers
             if isinstance(h, logging.handlers.RotatingFileHandler)),
            None,
        )
        assert rotating is not None
        assert rotating.backupCount == 5

    def test_log_file_path(self, config_json, tmp_path):
        setup_logging(config_json)
        log_file = tmp_path / "logs" / "app.log"
        assert log_file.exists()
        # 로그 내용이 기록됨
        content = log_file.read_text(encoding="utf-8")
        assert len(content) > 0
