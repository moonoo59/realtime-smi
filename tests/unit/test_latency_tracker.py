"""
LatencyTracker 단위 테스트

검증 조건:
- mock 데이터 100개 투입 시 P95 계산 오차 1ms 이하
- CSV 저장 파일 파싱 가능
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
import numpy as np
import pytest

from src.config.schema import AppConfig, MetricsConfig
from src.metrics import LatencyStats
from src.metrics.latency_tracker import LatencyTracker, _TimestampRecord


# =========================================================================
# 공통 픽스처
# =========================================================================

@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.metrics.latency_window_sec = 60
    cfg.metrics.metrics_output_dir = "/tmp/test_metrics"
    return cfg


@pytest.fixture
def tracker(config):
    return LatencyTracker(config)


def _ns_now() -> int:
    return time.time_ns()


# =========================================================================
# _TimestampRecord 단위 테스트
# =========================================================================

class TestTimestampRecord:
    def test_capture_to_send_ms_both_set(self):
        r = _TimestampRecord(packet_id=0)
        r.capture_ns = 1_000_000_000
        r.stt_send_ns = 1_050_000_000   # 50ms 후
        assert r.capture_to_send_ms() == pytest.approx(50.0)

    def test_capture_to_send_ms_missing_capture(self):
        r = _TimestampRecord(packet_id=0)
        r.stt_send_ns = 1_050_000_000
        assert r.capture_to_send_ms() is None

    def test_capture_to_send_ms_missing_send(self):
        r = _TimestampRecord(packet_id=0)
        r.capture_ns = 1_000_000_000
        assert r.capture_to_send_ms() is None

    def test_send_to_receive_ms(self):
        r = _TimestampRecord(packet_id=0)
        r.stt_send_ns = 1_000_000_000
        r.stt_receive_ns = 1_200_000_000  # 200ms 후
        assert r.send_to_receive_ms() == pytest.approx(200.0)

    def test_send_to_receive_ms_missing(self):
        r = _TimestampRecord(packet_id=0)
        assert r.send_to_receive_ms() is None

    def test_end_to_end_ms(self):
        r = _TimestampRecord(packet_id=0)
        r.capture_ns = 0
        r.display_ns = 300_000_000  # 300ms 후
        assert r.end_to_end_ms() == pytest.approx(300.0)

    def test_end_to_end_ms_missing(self):
        r = _TimestampRecord(packet_id=0)
        assert r.end_to_end_ms() is None


# =========================================================================
# LatencyTracker 초기화 테스트
# =========================================================================

class TestLatencyTrackerInit:
    def test_init_creates_empty_records(self, tracker):
        assert len(tracker._records) == 0

    def test_init_window_sec(self, tracker):
        assert tracker._window_sec == 60

    def test_init_output_dir(self, tracker):
        assert str(tracker._output_dir) == "/tmp/test_metrics"


# =========================================================================
# 타임스탬프 기록 테스트
# =========================================================================

class TestRecordMethods:
    def test_record_capture_creates_record(self, tracker):
        tracker.record_capture(packet_id=1, timestamp_ns=1_000_000_000)
        assert 1 in tracker._records
        assert tracker._records[1].capture_ns == 1_000_000_000

    def test_record_capture_idempotent_packet_id(self, tracker):
        """같은 packet_id를 두 번 기록하면 덮어씌워진다."""
        tracker.record_capture(packet_id=1, timestamp_ns=1_000_000_000)
        tracker.record_capture(packet_id=1, timestamp_ns=2_000_000_000)
        assert tracker._records[1].capture_ns == 2_000_000_000

    def test_record_stt_send_creates_record(self, tracker):
        tracker.record_stt_send(packet_id=5, timestamp_ns=1_000_000_000)
        assert 5 in tracker._records
        assert tracker._records[5].stt_send_ns == 1_000_000_000

    def test_record_stt_send_adds_capture_to_send_measurement(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=1, timestamp_ns=base)
        tracker.record_stt_send(packet_id=1, timestamp_ns=base + 50_000_000)  # +50ms
        measurements = tracker._measurements[LatencyTracker.STAGE_CAPTURE_TO_SEND]
        assert len(measurements) == 1
        assert measurements[0][1] == pytest.approx(50.0)

    def test_record_stt_send_no_measurement_without_capture(self, tracker):
        tracker.record_stt_send(packet_id=99, timestamp_ns=_ns_now())
        assert len(tracker._measurements[LatencyTracker.STAGE_CAPTURE_TO_SEND]) == 0

    def test_record_stt_receive_partial(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=1, timestamp_ns=base)
        tracker.record_stt_send(packet_id=1, timestamp_ns=base + 10_000_000)
        tracker.record_stt_receive(
            result_id=100, packet_id=1,
            timestamp_ns=base + 210_000_000,  # send 이후 200ms
            result_type="partial",
        )
        assert LatencyTracker.STAGE_SEND_TO_PARTIAL in tracker._measurements
        ms = tracker._measurements[LatencyTracker.STAGE_SEND_TO_PARTIAL][0][1]
        assert ms == pytest.approx(200.0)

    def test_record_stt_receive_final(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=2, timestamp_ns=base)
        tracker.record_stt_send(packet_id=2, timestamp_ns=base + 10_000_000)
        tracker.record_stt_receive(
            result_id=200, packet_id=2,
            timestamp_ns=base + 510_000_000,  # 500ms
            result_type="final",
        )
        assert LatencyTracker.STAGE_SEND_TO_FINAL in tracker._measurements
        ms = tracker._measurements[LatencyTracker.STAGE_SEND_TO_FINAL][0][1]
        assert ms == pytest.approx(500.0)

    def test_record_stt_receive_unknown_packet_skipped(self, tracker):
        """packet_id가 없으면 측정값을 추가하지 않는다."""
        tracker.record_stt_receive(
            result_id=999, packet_id=999,
            timestamp_ns=_ns_now(), result_type="final",
        )
        assert len(tracker._measurements[LatencyTracker.STAGE_SEND_TO_FINAL]) == 0

    def test_record_display_adds_e2e_measurement(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=3, timestamp_ns=base)
        tracker.record_stt_send(packet_id=3, timestamp_ns=base + 10_000_000)
        tracker.record_stt_receive(
            result_id=300, packet_id=3,
            timestamp_ns=base + 210_000_000, result_type="final",
        )
        tracker.record_display(result_id=300, timestamp_ns=base + 310_000_000)  # +300ms total
        ms_list = tracker._measurements[LatencyTracker.STAGE_END_TO_END]
        assert len(ms_list) == 1
        assert ms_list[0][1] == pytest.approx(310.0)

    def test_record_display_unknown_result_id_skipped(self, tracker):
        tracker.record_display(result_id=9999, timestamp_ns=_ns_now())
        assert len(tracker._measurements[LatencyTracker.STAGE_END_TO_END]) == 0


# =========================================================================
# P95 정확도 테스트 (핵심 검증 조건: 오차 1ms 이하)
# =========================================================================

class TestComputeStats:
    def test_p95_accuracy_100_samples(self, tracker):
        """
        100개 mock 데이터 투입 시 P95 계산 오차 1ms 이하.

        알려진 분포: 1~100ms 균등 분포 → numpy P95 = 95.05ms (기준값)
        """
        base = _ns_now()
        rng = np.random.default_rng(42)
        # 1~100ms 범위의 100개 지연 값 생성
        latencies_ms = rng.uniform(1.0, 100.0, 100)
        expected_p95 = float(np.percentile(latencies_ms, 95))

        for i, latency_ms in enumerate(latencies_ms):
            capture_ts = base + i * 1_000_000_000
            send_ts = capture_ts + int(latency_ms * 1_000_000)
            tracker.record_capture(packet_id=i, timestamp_ns=capture_ts)
            tracker.record_stt_send(packet_id=i, timestamp_ns=send_ts)

        stats = tracker.compute_stats()
        assert LatencyTracker.STAGE_CAPTURE_TO_SEND in stats
        s = stats[LatencyTracker.STAGE_CAPTURE_TO_SEND]
        assert abs(s.p95_ms - expected_p95) < 1.0, (
            f"P95 오차 초과: computed={s.p95_ms:.3f}ms, expected={expected_p95:.3f}ms"
        )

    def test_p99_accuracy_100_samples(self, tracker):
        """P99 계산 오차 1ms 이하."""
        base = _ns_now()
        rng = np.random.default_rng(7)
        latencies_ms = rng.uniform(10.0, 500.0, 100)
        expected_p99 = float(np.percentile(latencies_ms, 99))

        for i, lat in enumerate(latencies_ms):
            ts = base + i * 1_000_000_000
            tracker.record_capture(packet_id=i, timestamp_ns=ts)
            tracker.record_stt_send(packet_id=i, timestamp_ns=ts + int(lat * 1_000_000))

        stats = tracker.compute_stats()
        s = stats[LatencyTracker.STAGE_CAPTURE_TO_SEND]
        assert abs(s.p99_ms - expected_p99) < 1.0

    def test_compute_stats_count(self, tracker):
        base = _ns_now()
        for i in range(50):
            tracker.record_capture(packet_id=i, timestamp_ns=base + i * 10_000_000)
            tracker.record_stt_send(packet_id=i, timestamp_ns=base + i * 10_000_000 + 20_000_000)

        stats = tracker.compute_stats()
        assert stats[LatencyTracker.STAGE_CAPTURE_TO_SEND].count == 50

    def test_compute_stats_mean(self, tracker):
        base = _ns_now()
        # 정확히 100ms 고정 지연 10개
        for i in range(10):
            ts = base + i * 1_000_000_000
            tracker.record_capture(packet_id=i, timestamp_ns=ts)
            tracker.record_stt_send(packet_id=i, timestamp_ns=ts + 100_000_000)

        stats = tracker.compute_stats()
        s = stats[LatencyTracker.STAGE_CAPTURE_TO_SEND]
        assert s.mean_ms == pytest.approx(100.0)
        assert s.min_ms == pytest.approx(100.0)
        assert s.max_ms == pytest.approx(100.0)

    def test_compute_stats_empty_returns_empty(self, tracker):
        stats = tracker.compute_stats()
        assert stats == {}

    def test_compute_stats_window_filters_old(self, tracker):
        """윈도우 밖의 데이터는 통계에서 제외된다."""
        # 윈도우보다 훨씬 오래된 타임스탬프 (120초 전)
        old_ts = _ns_now() - 120 * 1_000_000_000
        tracker.record_capture(packet_id=0, timestamp_ns=old_ts)
        tracker.record_stt_send(packet_id=0, timestamp_ns=old_ts + 50_000_000)

        # 단: _add_measurement도 old_ts로 들어가므로 필터됨
        # compute_stats는 현재 시각 기준 60초 이내만 포함
        stats = tracker.compute_stats(window_sec=60)
        # 오래된 데이터는 필터링되어 stats가 비어야 함
        assert LatencyTracker.STAGE_CAPTURE_TO_SEND not in stats

    def test_compute_stats_custom_window(self, tracker):
        """window_sec 파라미터가 적용된다."""
        base = _ns_now()
        for i in range(5):
            ts = base + i * 1_000_000_000
            tracker.record_capture(packet_id=i, timestamp_ns=ts)
            tracker.record_stt_send(packet_id=i, timestamp_ns=ts + 30_000_000)

        stats = tracker.compute_stats(window_sec=300)
        assert stats[LatencyTracker.STAGE_CAPTURE_TO_SEND].count == 5

    def test_compute_stats_returns_latency_stats_type(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=0, timestamp_ns=base)
        tracker.record_stt_send(packet_id=0, timestamp_ns=base + 50_000_000)

        stats = tracker.compute_stats()
        assert isinstance(stats[LatencyTracker.STAGE_CAPTURE_TO_SEND], LatencyStats)

    def test_multiple_stages_in_stats(self, tracker):
        """여러 단계가 동시에 통계에 포함된다."""
        base = _ns_now()
        tracker.record_capture(packet_id=1, timestamp_ns=base)
        tracker.record_stt_send(packet_id=1, timestamp_ns=base + 20_000_000)
        tracker.record_stt_receive(
            result_id=1, packet_id=1,
            timestamp_ns=base + 220_000_000, result_type="partial",
        )
        tracker.record_stt_receive(
            result_id=2, packet_id=1,
            timestamp_ns=base + 420_000_000, result_type="final",
        )

        stats = tracker.compute_stats()
        assert LatencyTracker.STAGE_CAPTURE_TO_SEND in stats
        assert LatencyTracker.STAGE_SEND_TO_PARTIAL in stats
        assert LatencyTracker.STAGE_SEND_TO_FINAL in stats


# =========================================================================
# get_e2e_latency 테스트
# =========================================================================

class TestGetE2eLatency:
    def test_returns_none_for_unknown_result(self, tracker):
        assert tracker.get_e2e_latency(result_id=999) is None

    def test_returns_none_when_display_not_recorded(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=1, timestamp_ns=base)
        tracker.record_stt_receive(
            result_id=10, packet_id=1,
            timestamp_ns=base + 100_000_000, result_type="final",
        )
        assert tracker.get_e2e_latency(result_id=10) is None

    def test_returns_e2e_ms(self, tracker):
        base = _ns_now()
        tracker.record_capture(packet_id=1, timestamp_ns=base)
        tracker.record_stt_send(packet_id=1, timestamp_ns=base + 10_000_000)
        tracker.record_stt_receive(
            result_id=10, packet_id=1,
            timestamp_ns=base + 110_000_000, result_type="final",
        )
        tracker.record_display(result_id=10, timestamp_ns=base + 400_000_000)
        e2e = tracker.get_e2e_latency(result_id=10)
        assert e2e == pytest.approx(400.0)


# =========================================================================
# export_csv 테스트 (검증 조건: CSV 파일 파싱 가능)
# =========================================================================

class TestExportCsv:
    def test_csv_is_parseable(self, tracker, tmp_path):
        """CSV 저장 파일 파싱 가능 (헤더 + 데이터 행 포함)."""
        base = _ns_now()
        for i in range(5):
            ts = base + i * 1_000_000_000
            tracker.record_capture(packet_id=i, timestamp_ns=ts)
            tracker.record_stt_send(packet_id=i, timestamp_ns=ts + 30_000_000)
            tracker.record_stt_receive(
                result_id=i * 10, packet_id=i,
                timestamp_ns=ts + 230_000_000, result_type="final",
            )
            tracker.record_display(result_id=i * 10, timestamp_ns=ts + 330_000_000)

        filepath = tmp_path / "latency.csv"
        tracker.export_csv(filepath)

        assert filepath.exists()
        rows = list(csv.DictReader(filepath.open(encoding="utf-8")))
        assert len(rows) == 5

    def test_csv_header_columns(self, tracker, tmp_path):
        """CSV 헤더가 지정된 컬럼을 모두 포함한다."""
        tracker.record_capture(packet_id=0, timestamp_ns=_ns_now())
        filepath = tmp_path / "header_check.csv"
        tracker.export_csv(filepath)

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames

        expected_cols = [
            "packet_id", "capture_ns", "stt_send_ns", "stt_receive_ns",
            "display_ns", "result_type",
            "capture_to_send_ms", "send_to_receive_ms", "end_to_end_ms",
        ]
        for col in expected_cols:
            assert col in header, f"컬럼 누락: {col}"

    def test_csv_numeric_values(self, tracker, tmp_path):
        """CSV의 지연시간 값이 숫자로 파싱된다."""
        base = _ns_now()
        tracker.record_capture(packet_id=0, timestamp_ns=base)
        tracker.record_stt_send(packet_id=0, timestamp_ns=base + 75_000_000)  # 75ms

        filepath = tmp_path / "numeric.csv"
        tracker.export_csv(filepath)

        rows = list(csv.DictReader(filepath.open(encoding="utf-8")))
        assert len(rows) == 1
        val = float(rows[0]["capture_to_send_ms"])
        assert val == pytest.approx(75.0, abs=0.001)

    def test_csv_empty_when_no_timestamps(self, tracker, tmp_path):
        """타임스탬프 일부가 없으면 해당 컬럼이 빈 문자열이다."""
        tracker.record_capture(packet_id=0, timestamp_ns=_ns_now())
        # stt_send_ns 없음

        filepath = tmp_path / "empty_fields.csv"
        tracker.export_csv(filepath)

        rows = list(csv.DictReader(filepath.open(encoding="utf-8")))
        assert rows[0]["stt_send_ns"] == ""
        assert rows[0]["capture_to_send_ms"] == ""

    def test_csv_parent_dir_created(self, tracker, tmp_path):
        """중간 디렉토리가 없어도 자동 생성된다."""
        nested = tmp_path / "a" / "b" / "c" / "out.csv"
        tracker.record_capture(packet_id=0, timestamp_ns=_ns_now())
        tracker.export_csv(nested)
        assert nested.exists()

    def test_export_csv_100_records(self, tracker, tmp_path):
        """100개 레코드를 저장하고 파싱할 수 있다."""
        base = _ns_now()
        for i in range(100):
            ts = base + i * 1_000_000_000
            tracker.record_capture(packet_id=i, timestamp_ns=ts)
            tracker.record_stt_send(packet_id=i, timestamp_ns=ts + (i + 1) * 1_000_000)

        filepath = tmp_path / "hundred.csv"
        tracker.export_csv(filepath)

        rows = list(csv.DictReader(filepath.open(encoding="utf-8")))
        assert len(rows) == 100
        # 모든 capture_to_send_ms 값이 파싱 가능
        for row in rows:
            val = float(row["capture_to_send_ms"])
            assert val > 0


# =========================================================================
# 슬라이딩 윈도우 정리 테스트
# =========================================================================

class TestSlidingWindowCleanup:
    def test_old_measurements_are_cleaned(self, tracker):
        """_add_measurement는 윈도우 2배 이상 오래된 데이터를 제거한다."""
        stage = LatencyTracker.STAGE_CAPTURE_TO_SEND
        # 매우 오래된 타임스탬프 직접 삽입
        ancient_ts = _ns_now() - int(tracker._window_sec * 3 * 1_000_000_000)
        tracker._measurements[stage].append((ancient_ts, 10.0))
        assert len(tracker._measurements[stage]) == 1

        # 새 데이터 추가 → 정리 트리거
        fresh_ts = _ns_now()
        tracker._add_measurement(stage, fresh_ts, 20.0)

        remaining = tracker._measurements[stage]
        ts_list = [ts for ts, _ in remaining]
        assert ancient_ts not in ts_list
        assert fresh_ts in ts_list

    def test_recent_measurements_not_cleaned(self, tracker):
        """윈도우 내의 데이터는 정리되지 않는다."""
        stage = LatencyTracker.STAGE_CAPTURE_TO_SEND
        recent_ts = _ns_now() - 10_000_000_000  # 10초 전 (60초 윈도우 내)
        tracker._measurements[stage].append((recent_ts, 30.0))

        fresh_ts = _ns_now()
        tracker._add_measurement(stage, fresh_ts, 40.0)

        ts_list = [ts for ts, _ in tracker._measurements[stage]]
        assert recent_ts in ts_list
        assert fresh_ts in ts_list


# =========================================================================
# 스레드 안전성 테스트
# =========================================================================

class TestThreadSafety:
    def test_concurrent_record_capture(self, tracker):
        """멀티스레드 동시 기록 시 손실 없음."""
        import threading

        errors = []

        def record(start_id):
            try:
                for i in range(50):
                    tracker.record_capture(
                        packet_id=start_id + i,
                        timestamp_ns=_ns_now(),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(tracker._records) == 200

    def test_concurrent_read_write(self, tracker):
        """동시 읽기/쓰기 시 데드락 없음."""
        import threading

        stop_event = threading.Event()
        errors = []

        def writer():
            for i in range(100):
                try:
                    tracker.record_capture(packet_id=i, timestamp_ns=_ns_now())
                except Exception as e:
                    errors.append(e)

        def reader():
            while not stop_event.is_set():
                try:
                    tracker.compute_stats()
                except Exception as e:
                    errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        r.start()
        w.start()
        w.join()
        stop_event.set()
        r.join(timeout=2.0)

        assert not errors
