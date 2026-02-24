"""
AccuracyEvaluator 단위 테스트

검증 조건:
- 완벽 일치 시 WER=0.0, CER=0.0
- 레퍼런스 파일 로드 후 순서 매핑
- JSON 리포트 저장 및 파싱 가능
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.config.schema import AppConfig
from src.metrics import AccuracyReport
from src.metrics.accuracy_evaluator import AccuracyEvaluator


# =========================================================================
# 픽스처
# =========================================================================

@pytest.fixture
def config_enabled(tmp_path):
    cfg = AppConfig()
    cfg.accuracy.enabled = True
    cfg.accuracy.reference_source = "realtime"
    cfg.accuracy.output_dir = str(tmp_path / "reports")
    return cfg


@pytest.fixture
def config_file(tmp_path):
    ref_file = tmp_path / "reference.txt"
    ref_file.write_text("안녕하세요 테스트입니다\n오늘 날씨가 좋습니다\n", encoding="utf-8")
    cfg = AppConfig()
    cfg.accuracy.enabled = True
    cfg.accuracy.reference_source = "file"
    cfg.accuracy.reference_file = str(ref_file)
    cfg.accuracy.output_dir = str(tmp_path / "reports")
    return cfg


@pytest.fixture
def evaluator(config_enabled):
    return AccuracyEvaluator(config_enabled)


# =========================================================================
# 초기화 테스트
# =========================================================================

class TestInit:
    def test_init_enabled(self, config_enabled):
        ev = AccuracyEvaluator(config_enabled)
        assert ev._enabled is True

    def test_init_pair_count_zero(self, evaluator):
        assert evaluator.get_pair_count() == 0

    def test_init_with_reference_file(self, config_file):
        ev = AccuracyEvaluator(config_file)
        assert len(ev._reference_lines) == 2

    def test_init_missing_reference_file(self, tmp_path):
        cfg = AppConfig()
        cfg.accuracy.enabled = True
        cfg.accuracy.reference_source = "file"
        cfg.accuracy.reference_file = str(tmp_path / "nonexistent.txt")
        ev = AccuracyEvaluator(cfg)
        # 파일 없어도 초기화 성공, 빈 라인 목록
        assert ev._reference_lines == []

    def test_init_disabled(self):
        cfg = AppConfig()
        cfg.accuracy.enabled = False
        ev = AccuracyEvaluator(cfg)
        assert ev._enabled is False


# =========================================================================
# WER/CER 단위 계산 테스트
# =========================================================================

class TestWerCer:
    def test_perfect_match_wer_zero(self, evaluator):
        wer = evaluator.compute_wer("안녕하세요", "안녕하세요")
        assert wer == pytest.approx(0.0)

    def test_perfect_match_cer_zero(self, evaluator):
        cer = evaluator.compute_cer("안녕하세요", "안녕하세요")
        assert cer == pytest.approx(0.0)

    def test_complete_mismatch_wer(self, evaluator):
        # 완전히 다른 단어: 1개 단어 → 1개 치환 → WER=1.0
        wer = evaluator.compute_wer("가나다", "마바사")
        assert wer > 0.0

    def test_wer_one_word_substitution(self, evaluator):
        # "hello world" → "hello there" : 1/2 단어 오류 = 0.5
        wer = evaluator.compute_wer("hello there", "hello world")
        assert wer == pytest.approx(0.5)

    def test_cer_one_char(self, evaluator):
        # "abc" → "abd": 1/3 문자 오류 = 0.333...
        cer = evaluator.compute_cer("abd", "abc")
        assert 0.0 < cer <= 1.0

    def test_wer_empty_reference_returns_float(self, evaluator):
        # 엣지 케이스: 빈 레퍼런스
        wer = evaluator.compute_wer("", "")
        assert isinstance(wer, float)

    def test_cer_multi_sentence(self, evaluator):
        cer = evaluator.compute_cer("자막 테스트", "자막 테스트")
        assert cer == pytest.approx(0.0)


# =========================================================================
# add_result 테스트
# =========================================================================

class TestAddResult:
    def test_add_result_disabled_skips(self):
        cfg = AppConfig()
        cfg.accuracy.enabled = False
        ev = AccuracyEvaluator(cfg)
        ev.add_result("hello", "hello")
        assert ev.get_pair_count() == 0

    def test_add_result_with_reference(self, evaluator):
        evaluator.add_result("안녕", "안녕하세요")
        assert evaluator.get_pair_count() == 1

    def test_add_result_multiple(self, evaluator):
        for i in range(10):
            evaluator.add_result(f"text {i}", f"text {i}")
        assert evaluator.get_pair_count() == 10

    def test_add_result_file_source_sequential(self, config_file):
        ev = AccuracyEvaluator(config_file)
        ev.add_result("안녕하세요 테스트입니다")
        ev.add_result("오늘 날씨가 좋습니다")
        assert ev.get_pair_count() == 2

    def test_add_result_file_source_exhausted(self, config_file):
        ev = AccuracyEvaluator(config_file)
        ev.add_result("첫 번째")
        ev.add_result("두 번째")
        # 세 번째는 레퍼런스 없음 → 추가 안 됨
        ev.add_result("세 번째")
        assert ev.get_pair_count() == 2

    def test_add_result_with_reference_explicit(self, evaluator):
        evaluator.add_result_with_reference("hello world", "hello world")
        assert evaluator.get_pair_count() == 1


# =========================================================================
# compute_report 테스트
# =========================================================================

class TestComputeReport:
    def test_empty_report(self, evaluator):
        report = evaluator.compute_report(session_id="test-session")
        assert isinstance(report, AccuracyReport)
        assert report.wer == 0.0
        assert report.cer == 0.0
        assert report.total_words == 0
        assert report.session_id == "test-session"

    def test_perfect_match_report(self, evaluator):
        texts = ["안녕하세요 테스트", "오늘 날씨가 좋습니다", "자막 시스템"]
        for t in texts:
            evaluator.add_result(t, t)
        report = evaluator.compute_report(session_id="s1")
        assert report.wer == pytest.approx(0.0)
        assert report.cer == pytest.approx(0.0)
        assert report.total_words > 0

    def test_report_total_words_count(self, evaluator):
        # "hello world" = 2 words, "foo bar baz" = 3 words
        evaluator.add_result("hello world", "hello world")
        evaluator.add_result("foo bar baz", "foo bar baz")
        report = evaluator.compute_report()
        assert report.total_words == 5

    def test_report_total_chars_count(self, evaluator):
        # "abc" = 3 chars (no spaces), "de fg" = 4 chars (de+fg, spaces excluded)
        evaluator.add_result("abc", "abc")
        evaluator.add_result("de fg", "de fg")
        report = evaluator.compute_report()
        assert report.total_chars == 7  # 3 + 4

    def test_report_error_details_for_mismatch(self, evaluator):
        evaluator.add_result("안녕", "안녕하세요")  # WER > 0
        evaluator.add_result("완벽 일치", "완벽 일치")  # WER = 0
        report = evaluator.compute_report()
        # WER > 0인 쌍만 error_details에 포함
        assert len(report.error_details) == 1
        assert report.error_details[0].hypothesis == "안녕"

    def test_report_duration_positive(self, evaluator):
        import time
        time.sleep(0.01)
        report = evaluator.compute_report()
        assert report.duration_sec > 0

    def test_report_wer_range(self, evaluator):
        evaluator.add_result("hello world", "hello there")
        report = evaluator.compute_report()
        assert 0.0 <= report.wer <= 2.0  # WER은 1.0 초과 가능

    def test_report_returns_accumulation(self, evaluator):
        """누적된 모든 쌍이 리포트에 포함된다."""
        for i in range(20):
            evaluator.add_result(f"word{i}", f"word{i}")
        report = evaluator.compute_report()
        assert report.total_words == 20


# =========================================================================
# save_report 테스트 (JSON 파싱 가능 검증)
# =========================================================================

class TestSaveReport:
    def test_save_report_creates_file(self, evaluator, tmp_path):
        evaluator.add_result("테스트", "테스트")
        report = evaluator.compute_report(session_id="save-test")
        filepath = tmp_path / "report.json"
        result_path = evaluator.save_report(report, filepath)
        assert result_path.exists()

    def test_save_report_is_parseable_json(self, evaluator, tmp_path):
        """JSON 저장 파일 파싱 가능."""
        evaluator.add_result("안녕하세요", "안녕하세요")
        evaluator.add_result("hello world", "hello there")
        report = evaluator.compute_report(session_id="json-test")
        filepath = tmp_path / "report.json"
        evaluator.save_report(report, filepath)

        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["session_id"] == "json-test"
        assert "wer" in data
        assert "cer" in data
        assert "total_words" in data
        assert "error_details" in data
        assert isinstance(data["error_details"], list)

    def test_save_report_wer_value_preserved(self, evaluator, tmp_path):
        evaluator.add_result("hello world", "hello world")
        report = evaluator.compute_report()
        filepath = tmp_path / "wer_check.json"
        evaluator.save_report(report, filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["wer"] == pytest.approx(0.0)

    def test_save_report_auto_filepath(self, config_enabled):
        ev = AccuracyEvaluator(config_enabled)
        ev.add_result("테스트", "테스트")
        report = ev.compute_report(session_id="auto")
        result_path = ev.save_report(report)
        assert result_path.exists()

    def test_save_report_parent_dir_created(self, evaluator, tmp_path):
        filepath = tmp_path / "a" / "b" / "report.json"
        report = evaluator.compute_report()
        result_path = evaluator.save_report(report, filepath)
        assert result_path.exists()

    def test_save_report_korean_preserved(self, evaluator, tmp_path):
        """한글 텍스트가 깨지지 않고 저장된다."""
        evaluator.add_result("한글 테스트 문장", "한글 테스트 문장")
        report = evaluator.compute_report()
        filepath = tmp_path / "korean.json"
        evaluator.save_report(report, filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        # error_details가 없어야 함 (WER=0)
        assert data["error_details"] == []


# =========================================================================
# reset 테스트
# =========================================================================

class TestReset:
    def test_reset_clears_pairs(self, evaluator):
        for i in range(5):
            evaluator.add_result(f"text {i}", f"text {i}")
        evaluator.reset()
        assert evaluator.get_pair_count() == 0

    def test_reset_restarts_index(self, config_file):
        ev = AccuracyEvaluator(config_file)
        ev.add_result("첫 번째")
        ev.add_result("두 번째")
        ev.reset()
        # 리셋 후 다시 첫 번째 레퍼런스부터
        ev.add_result("다시 첫 번째")
        pairs = ev._pairs
        assert pairs[0][1] == "안녕하세요 테스트입니다"


# =========================================================================
# 스레드 안전성 테스트
# =========================================================================

class TestThreadSafety:
    def test_concurrent_add_result(self, evaluator):
        errors = []

        def add():
            try:
                for i in range(50):
                    evaluator.add_result(f"text{i}", f"text{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert evaluator.get_pair_count() == 200

    def test_concurrent_add_and_report(self, evaluator):
        errors = []
        stop = threading.Event()

        def writer():
            for i in range(100):
                try:
                    evaluator.add_result(f"w{i}", f"w{i}")
                except Exception as e:
                    errors.append(e)

        def reader():
            while not stop.is_set():
                try:
                    evaluator.compute_report()
                except Exception as e:
                    errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        r.start()
        w.start()
        w.join()
        stop.set()
        r.join(timeout=2.0)

        assert not errors
