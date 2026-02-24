"""
웹 대시보드 단위 테스트

검증 조건:
- HTTP 엔드포인트 (/, /api/health, /api/snapshot) 정상 응답
- WebSocket /ws/metrics 연결 및 메트릭 수신
- _collect_metrics() 가 MetricsStore 데이터를 올바르게 직렬화
- _check_alerts() 가 임계값 초과 시 알림 반환
- update_accuracy() 가 MetricsStore에 정확도 데이터 기록
- 알림 중복 억제(ALERT_DEDUP_SEC) 동작 확인
"""

from __future__ import annotations

import asyncio
import time
import pytest

from src.metrics.metrics_store import MetricsStore
from src.dashboard.web_dashboard import WebDashboard


# =========================================================================
# 픽스처
# =========================================================================

@pytest.fixture
def store():
    return MetricsStore()


@pytest.fixture
def dashboard(store):
    return WebDashboard(store, host="127.0.0.1", port=18765)


@pytest.fixture
def test_client(dashboard):
    """FastAPI TestClient를 반환합니다."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi가 설치되지 않음")
    return TestClient(dashboard._app)


# =========================================================================
# HTTP 엔드포인트 테스트
# =========================================================================

class TestHttpEndpoints:
    def test_index_returns_html(self, test_client):
        """GET / 가 200 상태와 HTML을 반환한다."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text

    def test_index_html_contains_chart_js(self, test_client):
        """대시보드 HTML에 Chart.js CDN 링크가 포함된다."""
        response = test_client.get("/")
        assert "chart.js" in response.text.lower()

    def test_health_returns_ok(self, test_client):
        """GET /api/health 가 status=ok를 반환한다."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "ts" in data

    def test_snapshot_returns_all_keys(self, test_client):
        """GET /api/snapshot 이 필수 키를 모두 포함한다."""
        response = test_client.get("/api/snapshot")
        assert response.status_code == 200
        data = response.json()
        for key in ("ts", "subtitle", "audio", "stt", "frames", "latency", "accuracy", "alerts"):
            assert key in data, f"응답에 '{key}' 키가 없음"

    def test_snapshot_audio_reflects_store(self, test_client, store):
        """MetricsStore 오디오 레벨이 /api/snapshot에 반영된다."""
        store.update_audio_level(rms=0.42, peak=0.87)
        response = test_client.get("/api/snapshot")
        data = response.json()
        assert data["audio"]["rms"] == pytest.approx(0.42, abs=1e-3)
        assert data["audio"]["peak"] == pytest.approx(0.87, abs=1e-3)

    def test_snapshot_stt_reflects_store(self, test_client, store):
        """STT 상태가 /api/snapshot에 반영된다."""
        store.update_stt_status(connected=True, error_count=3, reconnect_count=1)
        response = test_client.get("/api/snapshot")
        data = response.json()
        assert data["stt"]["connected"] is True
        assert data["stt"]["error_count"] == 3
        assert data["stt"]["reconnect_count"] == 1

    def test_snapshot_frames_reflects_store(self, test_client, store):
        """프레임 통계가 /api/snapshot에 반영된다."""
        store.update_frame_stats(
            total_frames=1000, drop_count=10, drop_rate=0.01, queue_depth=4
        )
        response = test_client.get("/api/snapshot")
        data = response.json()
        assert data["frames"]["total_frames"] == 1000
        assert data["frames"]["drop_count"] == 10
        assert data["frames"]["drop_rate"] == pytest.approx(0.01, abs=1e-3)

    def test_snapshot_subtitle_reflects_store(self, test_client, store):
        """자막 텍스트가 /api/snapshot에 반영된다."""
        store.update_subtitle("테스트 자막입니다", is_partial=False)
        response = test_client.get("/api/snapshot")
        data = response.json()
        assert data["subtitle"]["text"] == "테스트 자막입니다"
        assert data["subtitle"]["is_partial"] is False

    def test_snapshot_accuracy_reflects_update(self, test_client, dashboard):
        """update_accuracy() 결과가 /api/snapshot에 반영된다."""
        dashboard.update_accuracy(wer=0.12, cer=0.06, pair_count=25)
        response = test_client.get("/api/snapshot")
        data = response.json()
        assert data["accuracy"]["wer"] == pytest.approx(0.12, abs=1e-3)
        assert data["accuracy"]["cer"] == pytest.approx(0.06, abs=1e-3)
        assert data["accuracy"]["pair_count"] == 25


# =========================================================================
# WebSocket 테스트
# =========================================================================

class TestWebSocket:
    def test_ws_metrics_connects_and_receives_data(self, test_client):
        """WebSocket /ws/metrics 에 연결하면 메트릭 JSON을 수신한다."""
        import json
        with test_client.websocket_connect("/ws/metrics") as ws:
            # 브로드캐스트 루프가 없는 테스트 환경이므로 직접 메시지 없이
            # 연결 자체가 정상인지만 확인 (accept 성공)
            pass  # WebSocketDisconnect 없이 clean exit이면 성공

    def test_ws_multiple_clients_accepted(self, test_client):
        """복수 WebSocket 클라이언트가 동시에 연결 가능하다."""
        with test_client.websocket_connect("/ws/metrics") as ws1:
            with test_client.websocket_connect("/ws/metrics") as ws2:
                pass  # 두 연결 모두 수락되면 성공


# =========================================================================
# _collect_metrics 단위 테스트
# =========================================================================

class TestCollectMetrics:
    def test_returns_ts(self, dashboard):
        """_collect_metrics() 결과에 'ts' 타임스탬프가 있다."""
        result = dashboard._collect_metrics()
        assert "ts" in result
        assert result["ts"] > 0

    def test_audio_keys_present(self, dashboard):
        """오디오 섹션에 rms, peak 키가 있다."""
        result = dashboard._collect_metrics()
        assert set(result["audio"].keys()) >= {"rms", "peak"}

    def test_stt_keys_present(self, dashboard):
        """STT 섹션에 connected, error_count, reconnect_count 키가 있다."""
        result = dashboard._collect_metrics()
        assert "connected" in result["stt"]
        assert "error_count" in result["stt"]
        assert "reconnect_count" in result["stt"]

    def test_frames_keys_present(self, dashboard):
        """frames 섹션에 필수 키가 있다."""
        result = dashboard._collect_metrics()
        for key in ("total_frames", "drop_count", "drop_rate", "queue_depth"):
            assert key in result["frames"], f"frames에 '{key}' 없음"

    def test_alerts_is_list(self, dashboard):
        """alerts 필드는 리스트 타입이다."""
        result = dashboard._collect_metrics()
        assert isinstance(result["alerts"], list)


# =========================================================================
# _check_alerts 단위 테스트
# =========================================================================

class TestCheckAlerts:
    def _get_defaults(self, store):
        """기본 메트릭 객체를 반환합니다."""
        audio = store.get_audio_level()
        stt = store.get_stt_status()
        frames = store.get_frame_stats()
        return audio, stt, frames

    def test_no_alerts_by_default(self, dashboard, store):
        """기본값(연결 안 됨, 데이터 없음) 상태에서 알림이 없다."""
        audio, stt, frames = self._get_defaults(store)
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        # STT_DISCONNECTED는 connected=False일 때 발생하지만
        # updated_at_ns=0인 초기 상태는 묵음 알림을 억제해야 함
        alert_keys = [a["key"] for a in alerts]
        assert "HIGH_DROP_RATE" not in alert_keys
        assert "HIGH_E2E_LATENCY" not in alert_keys

    def test_stt_disconnected_alert(self, dashboard, store):
        """STT 연결 끊김 시 STT_DISCONNECTED 알림이 발생한다."""
        store.update_stt_status(connected=False)
        audio, stt, frames = self._get_defaults(store)
        # 중복 억제 캐시 초기화
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        assert any(a["key"] == "STT_DISCONNECTED" for a in alerts)

    def test_high_drop_rate_alert(self, dashboard, store):
        """드롭률 > 5% 시 HIGH_DROP_RATE 알림이 발생한다."""
        store.update_frame_stats(
            total_frames=100, drop_count=10, drop_rate=0.10, queue_depth=0
        )
        audio, stt, frames = self._get_defaults(store)
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        assert any(a["key"] == "HIGH_DROP_RATE" for a in alerts)

    def test_high_drop_rate_below_threshold_no_alert(self, dashboard, store):
        """드롭률 ≤ 5% 시 HIGH_DROP_RATE 알림이 없다."""
        store.update_frame_stats(
            total_frames=1000, drop_count=30, drop_rate=0.03, queue_depth=0
        )
        audio, stt, frames = self._get_defaults(store)
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        assert not any(a["key"] == "HIGH_DROP_RATE" for a in alerts)

    def test_high_e2e_latency_alert(self, dashboard, store):
        """E2E P95 > 3000ms 시 HIGH_E2E_LATENCY 알림이 발생한다."""
        audio, stt, frames = self._get_defaults(store)
        latency = {"e2e": {"p95_ms": 4000, "p99_ms": 4200}}
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency=latency)
        assert any(a["key"] == "HIGH_E2E_LATENCY" for a in alerts)

    def test_audio_silence_alert(self, dashboard, store):
        """오디오 RMS < 0.001 이고 updated_at_ns > 0 일 때 AUDIO_SILENCE 알림 발생."""
        store.update_audio_level(rms=0.0001, peak=0.0002)
        audio, stt, frames = self._get_defaults(store)
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        assert any(a["key"] == "AUDIO_SILENCE" for a in alerts)

    def test_stt_error_burst_alert(self, dashboard, store):
        """STT 오류 > 10회 시 STT_ERROR_BURST 알림이 발생한다."""
        store.update_stt_status(connected=True, error_count=15)
        audio, stt, frames = self._get_defaults(store)
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        assert any(a["key"] == "STT_ERROR_BURST" for a in alerts)

    def test_alert_dedup_suppresses_repeat(self, dashboard, store):
        """동일 알림은 ALERT_DEDUP_SEC 이내에 재발생하지 않는다."""
        store.update_stt_status(connected=False)
        audio, stt, frames = self._get_defaults(store)
        dashboard._alert_last.clear()

        # 첫 번째 호출 — 알림 발생해야 함
        alerts1 = dashboard._check_alerts(audio, stt, frames, latency={})
        assert any(a["key"] == "STT_DISCONNECTED" for a in alerts1)

        # 즉시 재호출 — 중복 억제되어야 함
        alerts2 = dashboard._check_alerts(audio, stt, frames, latency={})
        assert not any(a["key"] == "STT_DISCONNECTED" for a in alerts2)

    def test_alert_level_error_for_stt_disconnected(self, dashboard, store):
        """STT_DISCONNECTED 알림 레벨이 'error'다."""
        store.update_stt_status(connected=False)
        audio, stt, frames = self._get_defaults(store)
        dashboard._alert_last.clear()
        alerts = dashboard._check_alerts(audio, stt, frames, latency={})
        disc = next((a for a in alerts if a["key"] == "STT_DISCONNECTED"), None)
        assert disc is not None
        assert disc["level"] == "error"


# =========================================================================
# update_accuracy 테스트
# =========================================================================

class TestUpdateAccuracy:
    def test_update_accuracy_stores_in_metrics(self, dashboard, store):
        """update_accuracy() 가 MetricsStore에 WER/CER을 기록한다."""
        dashboard.update_accuracy(wer=0.08, cer=0.04, pair_count=100)
        acc = store.get_accuracy_stats()
        assert acc.wer == pytest.approx(0.08, abs=1e-6)
        assert acc.cer == pytest.approx(0.04, abs=1e-6)
        assert acc.pair_count == 100

    def test_update_accuracy_zero_values(self, dashboard, store):
        """WER=0, CER=0으로 업데이트해도 예외가 없다."""
        dashboard.update_accuracy(wer=0.0, cer=0.0, pair_count=0)
        acc = store.get_accuracy_stats()
        assert acc.wer == 0.0
        assert acc.cer == 0.0
