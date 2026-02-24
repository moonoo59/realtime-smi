"""
TUI 대시보드 단위 테스트

검증 조건:
- Textual Pilot으로 앱 초기화 확인
- MetricsStore 변경 시 패널 갱신 확인
- 폴링 타이머 동작 확인
"""

from __future__ import annotations

import asyncio
import pytest

from src.config.schema import AppConfig
from src.metrics.metrics_store import MetricsStore
from src.dashboard.tui_dashboard import (
    TuiDashboard,
    SubtitlePanel,
    AudioLevelPanel,
    LatencyPanel,
    SttStatusPanel,
    FrameStatsPanel,
    AccuracyPanel,
)


# =========================================================================
# 픽스처
# =========================================================================

@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.dashboard.refresh_interval_ms = 500
    return cfg


@pytest.fixture
def store():
    return MetricsStore()


@pytest.fixture
def app(config, store):
    return TuiDashboard(config=config, metrics_store=store)


# =========================================================================
# 패널 위젯 단위 테스트
# =========================================================================

class TestSubtitlePanel:
    def test_update_partial(self):
        panel = SubtitlePanel()
        panel.update_subtitle("안녕하세요", is_partial=True)
        assert panel.text == "안녕하세요"
        assert panel.is_partial is True

    def test_update_final(self):
        panel = SubtitlePanel()
        panel.update_subtitle("완료 자막", is_partial=False)
        assert panel.is_partial is False

    def test_render_empty_shows_none(self):
        panel = SubtitlePanel()
        output = panel.render()
        assert "(없음)" in output

    def test_render_partial_label(self):
        panel = SubtitlePanel()
        panel.update_subtitle("테스트", is_partial=True)
        output = panel.render()
        assert "(중간)" in output

    def test_render_final_label(self):
        panel = SubtitlePanel()
        panel.update_subtitle("완료", is_partial=False)
        output = panel.render()
        assert "(확정)" in output


class TestAudioLevelPanel:
    def test_update_level(self):
        panel = AudioLevelPanel()
        panel.update_level(0.5, 0.8)
        assert panel.rms == pytest.approx(0.5)
        assert panel.peak == pytest.approx(0.8)

    def test_level_clamped_max(self):
        panel = AudioLevelPanel()
        panel.update_level(1.5, 2.0)
        assert panel.rms <= 1.0
        assert panel.peak <= 1.0

    def test_level_clamped_min(self):
        panel = AudioLevelPanel()
        panel.update_level(-0.1, -0.5)
        assert panel.rms >= 0.0
        assert panel.peak >= 0.0

    def test_render_contains_values(self):
        panel = AudioLevelPanel()
        panel.update_level(0.3, 0.6)
        output = panel.render()
        assert "RMS" in output
        assert "Peak" in output


class TestLatencyPanel:
    def test_update_empty_stats(self):
        panel = LatencyPanel()
        panel.update_stats({})
        assert "(데이터 없음)" in panel.stats_text

    def test_update_stats_with_data(self):
        from src.metrics import LatencyStats
        panel = LatencyPanel()
        stats = {
            "end_to_end": LatencyStats(
                stage="end_to_end",
                count=10,
                mean_ms=150.0,
                min_ms=100.0,
                max_ms=200.0,
                p95_ms=190.0,
                p99_ms=198.0,
            )
        }
        panel.update_stats(stats)
        assert "E2E" in panel.stats_text
        assert "150" in panel.stats_text

    def test_render_includes_latency_label(self):
        panel = LatencyPanel()
        output = panel.render()
        assert "[지연시간]" in output


class TestSttStatusPanel:
    def test_update_connected(self):
        panel = SttStatusPanel()
        panel.update_status(connected=True, error_count=0, reconnect_count=0)
        assert panel.connected is True

    def test_update_disconnected(self):
        panel = SttStatusPanel()
        panel.update_status(connected=False, error_count=3, reconnect_count=2)
        assert panel.connected is False
        assert panel.error_count == 3
        assert panel.reconnect_count == 2

    def test_render_connected_status(self):
        panel = SttStatusPanel()
        panel.update_status(True, 0, 0)
        output = panel.render()
        assert "연결됨" in output

    def test_render_disconnected_status(self):
        panel = SttStatusPanel()
        panel.update_status(False, 1, 0)
        output = panel.render()
        assert "끊김" in output


class TestFrameStatsPanel:
    def test_update_stats(self):
        panel = FrameStatsPanel()
        panel.update_frame_stats(
            total=1000, drop_count=10, drop_rate=0.01, queue_depth=5
        )
        assert panel.total_frames == 1000
        assert panel.drop_count == 10
        assert panel.drop_rate == pytest.approx(0.01)
        assert panel.queue_depth == 5

    def test_render_contains_frame_label(self):
        panel = FrameStatsPanel()
        output = panel.render()
        assert "[프레임]" in output


class TestAccuracyPanel:
    def test_update_accuracy(self):
        panel = AccuracyPanel()
        panel.update_accuracy(wer=0.1, cer=0.05, pair_count=50)
        assert panel.wer == pytest.approx(0.1)
        assert panel.cer == pytest.approx(0.05)
        assert panel.pair_count == 50

    def test_render_contains_wer_cer(self):
        panel = AccuracyPanel()
        panel.update_accuracy(0.2, 0.1, 10)
        output = panel.render()
        assert "WER" in output
        assert "CER" in output


# =========================================================================
# TuiDashboard 통합 테스트 (Textual Pilot 사용)
# =========================================================================

class TestTuiDashboard:
    @pytest.mark.asyncio
    async def test_app_starts_and_has_panels(self, app):
        """앱이 정상적으로 시작되고 6개 패널이 존재한다."""
        async with app.run_test() as pilot:
            # 각 패널 존재 확인
            assert app.query_one("#subtitle", SubtitlePanel) is not None
            assert app.query_one("#audio_level", AudioLevelPanel) is not None
            assert app.query_one("#latency", LatencyPanel) is not None
            assert app.query_one("#stt_status", SttStatusPanel) is not None
            assert app.query_one("#frame_stats", FrameStatsPanel) is not None
            assert app.query_one("#accuracy", AccuracyPanel) is not None

    @pytest.mark.asyncio
    async def test_subtitle_panel_updates_from_store(self, app, store):
        """MetricsStore 자막 업데이트가 패널에 반영된다."""
        async with app.run_test() as pilot:
            store.update_subtitle("안녕하세요 반갑습니다", is_partial=False)
            # 폴링 간격 대기
            await pilot.pause(0.6)
            app._poll_metrics()
            panel = app.query_one("#subtitle", SubtitlePanel)
            assert panel.text == "안녕하세요 반갑습니다"

    @pytest.mark.asyncio
    async def test_audio_panel_updates_from_store(self, app, store):
        """MetricsStore 오디오 레벨 업데이트가 패널에 반영된다."""
        async with app.run_test() as pilot:
            store.update_audio_level(rms=0.4, peak=0.7)
            app._poll_metrics()
            panel = app.query_one("#audio_level", AudioLevelPanel)
            assert panel.rms == pytest.approx(0.4)
            assert panel.peak == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_stt_panel_updates_from_store(self, app, store):
        """STT 상태 업데이트가 패널에 반영된다."""
        async with app.run_test() as pilot:
            store.update_stt_status(connected=True, error_count=2, reconnect_count=1)
            app._poll_metrics()
            panel = app.query_one("#stt_status", SttStatusPanel)
            assert panel.connected is True
            assert panel.error_count == 2

    @pytest.mark.asyncio
    async def test_frame_stats_panel_updates(self, app, store):
        """프레임 통계 업데이트가 패널에 반영된다."""
        async with app.run_test() as pilot:
            store.update_frame_stats(
                total_frames=500, drop_count=5, drop_rate=0.01, queue_depth=3
            )
            app._poll_metrics()
            panel = app.query_one("#frame_stats", FrameStatsPanel)
            assert panel.total_frames == 500
            assert panel.queue_depth == 3

    @pytest.mark.asyncio
    async def test_accuracy_panel_updates(self, app):
        """update_accuracy 직접 호출 시 패널이 갱신된다."""
        async with app.run_test() as pilot:
            app.update_accuracy(wer=0.15, cer=0.08, pair_count=30)
            panel = app.query_one("#accuracy", AccuracyPanel)
            assert panel.wer == pytest.approx(0.15)
            assert panel.pair_count == 30

    @pytest.mark.asyncio
    async def test_latency_panel_updates_from_store(self, app, store):
        """지연시간 통계가 패널에 반영된다."""
        from src.metrics import LatencyStats

        async with app.run_test() as pilot:
            store.update_latency_stats(
                "end_to_end",
                LatencyStats(
                    stage="end_to_end",
                    count=20,
                    mean_ms=200.0,
                    min_ms=100.0,
                    max_ms=350.0,
                    p95_ms=320.0,
                    p99_ms=345.0,
                ),
            )
            app._poll_metrics()
            panel = app.query_one("#latency", LatencyPanel)
            assert "E2E" in panel.stats_text

    @pytest.mark.asyncio
    async def test_refresh_interval_set(self, app):
        """갱신 주기가 설정값과 일치한다."""
        async with app.run_test() as pilot:
            assert app._refresh_interval_ms == 500
