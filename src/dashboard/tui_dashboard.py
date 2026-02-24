"""
Textual 기반 TUI 대시보드 모듈입니다.

역할:
- 6개 패널로 파이프라인 실시간 상태 표시:
  1. 현재 자막 (partial/final 구분)
  2. 오디오 레벨 미터 (RMS/Peak)
  3. 지연시간 통계 (P95/P99 per stage)
  4. STT 연결 상태 (connected, 오류/재연결 횟수)
  5. 프레임 통계 (총 프레임, 드롭율, 큐 깊이)
  6. WER/CER 정확도
- MetricsStore를 500ms 주기로 폴링

사용 예시:
    >>> dashboard = TuiDashboard(config, metrics_store)
    >>> await dashboard.run_async()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.reactive import reactive
from textual import work

from src.config.schema import AppConfig
from src.metrics.metrics_store import MetricsStore

logger = logging.getLogger(__name__)


# =========================================================================
# 패널 위젯
# =========================================================================

class SubtitlePanel(Static):
    """현재 자막 패널입니다."""

    DEFAULT_CSS = """
    SubtitlePanel {
        border: solid $primary;
        padding: 1 2;
        height: 5;
        margin: 0 1 1 1;
    }
    """

    text: reactive[str] = reactive("")
    is_partial: reactive[bool] = reactive(False)

    def render(self) -> str:
        label = "[자막] "
        status = "(중간)" if self.is_partial else "(확정)"
        content = self.text or "(없음)"
        return f"{label}{status} {content}"

    def update_subtitle(self, text: str, is_partial: bool) -> None:
        self.text = text
        self.is_partial = is_partial


class AudioLevelPanel(Static):
    """오디오 레벨 미터 패널입니다."""

    DEFAULT_CSS = """
    AudioLevelPanel {
        border: solid $secondary;
        padding: 1 2;
        height: 5;
        margin: 0 1 1 1;
    }
    """

    rms: reactive[float] = reactive(0.0)
    peak: reactive[float] = reactive(0.0)

    def render(self) -> str:
        bar_width = 30
        rms_bar = int(self.rms * bar_width)
        peak_bar = int(self.peak * bar_width)
        rms_str = "█" * rms_bar + "░" * (bar_width - rms_bar)
        peak_str = "█" * peak_bar + "░" * (bar_width - peak_bar)
        return (
            f"[오디오] RMS  [{rms_str}] {self.rms:.3f}\n"
            f"         Peak [{peak_str}] {self.peak:.3f}"
        )

    def update_level(self, rms: float, peak: float) -> None:
        self.rms = min(1.0, max(0.0, rms))
        self.peak = min(1.0, max(0.0, peak))


class LatencyPanel(Static):
    """지연시간 통계 패널입니다."""

    DEFAULT_CSS = """
    LatencyPanel {
        border: solid $warning;
        padding: 1 2;
        height: 8;
        margin: 0 1 1 1;
    }
    """

    stats_text: reactive[str] = reactive("(데이터 없음)")

    def render(self) -> str:
        return f"[지연시간]\n{self.stats_text}"

    def update_stats(self, stats: dict) -> None:
        if not stats:
            self.stats_text = "(데이터 없음)"
            return
        lines = []
        for stage, s in stats.items():
            short = stage.replace("capture_to_stt_send", "cap→send")
            short = short.replace("stt_send_to_partial", "send→part")
            short = short.replace("stt_send_to_final", "send→final")
            short = short.replace("end_to_end", "E2E")
            lines.append(
                f"{short:12s}: avg={s.mean_ms:6.0f}ms "
                f"P95={s.p95_ms:6.0f}ms P99={s.p99_ms:6.0f}ms (n={s.count})"
            )
        self.stats_text = "\n".join(lines)


class SttStatusPanel(Static):
    """STT 연결 상태 패널입니다."""

    DEFAULT_CSS = """
    SttStatusPanel {
        border: solid $success;
        padding: 1 2;
        height: 6;
        margin: 0 1 1 1;
    }
    """

    connected: reactive[bool] = reactive(False)
    error_count: reactive[int] = reactive(0)
    reconnect_count: reactive[int] = reactive(0)

    def render(self) -> str:
        status = "✓ 연결됨" if self.connected else "✗ 끊김"
        return (
            f"[STT 상태] {status}\n"
            f"오류: {self.error_count}회  "
            f"재연결: {self.reconnect_count}회"
        )

    def update_status(self, connected: bool, error_count: int, reconnect_count: int) -> None:
        self.connected = connected
        self.error_count = error_count
        self.reconnect_count = reconnect_count


class FrameStatsPanel(Static):
    """비디오 프레임 통계 패널입니다."""

    DEFAULT_CSS = """
    FrameStatsPanel {
        border: solid $accent;
        padding: 1 2;
        height: 6;
        margin: 0 1 1 1;
    }
    """

    total_frames: reactive[int] = reactive(0)
    drop_count: reactive[int] = reactive(0)
    drop_rate: reactive[float] = reactive(0.0)
    queue_depth: reactive[int] = reactive(0)

    def render(self) -> str:
        return (
            f"[프레임]\n"
            f"총: {self.total_frames}  드롭: {self.drop_count} "
            f"({self.drop_rate:.1%})  큐: {self.queue_depth}"
        )

    def update_frame_stats(
        self, total: int, drop_count: int, drop_rate: float, queue_depth: int
    ) -> None:
        self.total_frames = total
        self.drop_count = drop_count
        self.drop_rate = drop_rate
        self.queue_depth = queue_depth


class AccuracyPanel(Static):
    """WER/CER 정확도 패널입니다."""

    DEFAULT_CSS = """
    AccuracyPanel {
        border: solid $error;
        padding: 1 2;
        height: 5;
        margin: 0 1 1 1;
    }
    """

    wer: reactive[float] = reactive(0.0)
    cer: reactive[float] = reactive(0.0)
    pair_count: reactive[int] = reactive(0)

    def render(self) -> str:
        return (
            f"[정확도] WER={self.wer:.3f}  "
            f"CER={self.cer:.3f}  "
            f"샘플={self.pair_count}"
        )

    def update_accuracy(self, wer: float, cer: float, pair_count: int) -> None:
        self.wer = wer
        self.cer = cer
        self.pair_count = pair_count


# =========================================================================
# 메인 TUI 앱
# =========================================================================

class TuiDashboard(App):
    """
    SDI-RealtimeSubtitle 파이프라인 상태를 표시하는 TUI 대시보드입니다.

    MetricsStore를 500ms 주기로 폴링하여 6개 패널을 갱신합니다.
    """

    TITLE = "SDI-RealtimeSubtitle Dashboard"

    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(
        self,
        config: AppConfig,
        metrics_store: MetricsStore,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._metrics_store = metrics_store
        self._refresh_interval_ms = config.dashboard.refresh_interval_ms

    # =========================================================================
    # UI 구성
    # =========================================================================

    def compose(self) -> ComposeResult:
        yield Header()
        yield SubtitlePanel(id="subtitle")
        yield AudioLevelPanel(id="audio_level")
        yield LatencyPanel(id="latency")
        yield SttStatusPanel(id="stt_status")
        yield FrameStatsPanel(id="frame_stats")
        yield AccuracyPanel(id="accuracy")
        yield Footer()

    def on_mount(self) -> None:
        """앱 마운트 시 폴링 타이머를 시작합니다."""
        interval_sec = self._refresh_interval_ms / 1000.0
        self.set_interval(interval_sec, self._poll_metrics)
        logger.info(f"TUI 대시보드 시작: 갱신 주기={self._refresh_interval_ms}ms")

    # =========================================================================
    # 메트릭 폴링
    # =========================================================================

    def _poll_metrics(self) -> None:
        """MetricsStore에서 최신 메트릭을 읽어 패널을 갱신합니다."""
        try:
            self._update_subtitle()
            self._update_audio_level()
            self._update_latency()
            self._update_stt_status()
            self._update_frame_stats()
        except Exception as exc:
            logger.error(f"메트릭 폴링 오류: {exc}")

    def _update_subtitle(self) -> None:
        sub = self._metrics_store.get_current_subtitle()
        panel: SubtitlePanel = self.query_one("#subtitle", SubtitlePanel)
        panel.update_subtitle(sub.text, sub.is_partial)

    def _update_audio_level(self) -> None:
        level = self._metrics_store.get_audio_level()
        panel: AudioLevelPanel = self.query_one("#audio_level", AudioLevelPanel)
        panel.update_level(level.rms, level.peak)

    def _update_latency(self) -> None:
        stats = self._metrics_store.get_latency_stats()
        panel: LatencyPanel = self.query_one("#latency", LatencyPanel)
        panel.update_stats(stats)

    def _update_stt_status(self) -> None:
        status = self._metrics_store.get_stt_status()
        panel: SttStatusPanel = self.query_one("#stt_status", SttStatusPanel)
        panel.update_status(
            status.connected,
            status.error_count,
            status.reconnect_count,
        )

    def _update_frame_stats(self) -> None:
        fs = self._metrics_store.get_frame_stats()
        panel: FrameStatsPanel = self.query_one("#frame_stats", FrameStatsPanel)
        panel.update_frame_stats(
            fs.total_frames,
            fs.drop_count,
            fs.drop_rate,
            fs.queue_depth,
        )

    def update_accuracy(self, wer: float, cer: float, pair_count: int) -> None:
        """AccuracyEvaluator에서 직접 호출하여 정확도 패널을 갱신합니다."""
        panel: AccuracyPanel = self.query_one("#accuracy", AccuracyPanel)
        panel.update_accuracy(wer, cer, pair_count)
