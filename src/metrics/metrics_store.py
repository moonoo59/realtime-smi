"""
공유 메트릭 저장소 모듈입니다.

역할:
- 모든 모듈이 공유하는 thread-safe 메트릭 저장소
- 오디오 레벨(RMS/Peak), 지연시간 통계, STT 상태, 프레임 통계를 중앙 관리
- 대시보드가 폴링하여 현재 파이프라인 상태를 표시

사용 예시:
    >>> store = MetricsStore()
    >>> store.update_audio_level(rms=0.3, peak=0.7)
    >>> level = store.get_audio_level()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioLevel:
    """오디오 레벨 메트릭입니다."""
    rms: float = 0.0
    peak: float = 0.0
    updated_at_ns: int = 0


@dataclass
class STTStatus:
    """STT 연결 상태 메트릭입니다."""
    connected: bool = False
    error_count: int = 0
    reconnect_count: int = 0
    last_result_at_ns: int = 0
    updated_at_ns: int = 0


@dataclass
class FrameStats:
    """비디오 프레임 통계입니다."""
    total_frames: int = 0
    drop_count: int = 0
    drop_rate: float = 0.0
    queue_depth: int = 0
    updated_at_ns: int = 0


@dataclass
class CurrentSubtitle:
    """현재 표시 중인 자막 정보입니다."""
    text: str = ""
    is_partial: bool = False
    updated_at_ns: int = 0


@dataclass
class AccuracyStats:
    """WER/CER 정확도 통계입니다."""
    wer: float = 0.0
    cer: float = 0.0
    pair_count: int = 0
    updated_at_ns: int = 0


class MetricsStore:
    """
    파이프라인 전체 메트릭을 중앙에서 관리하는 thread-safe 저장소입니다.

    모든 공개 메서드는 RLock으로 보호되어 멀티스레드 환경에서 안전합니다.
    대시보드는 이 저장소를 주기적으로 폴링합니다.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._audio_level = AudioLevel()
        self._stt_status = STTStatus()
        self._frame_stats = FrameStats()
        self._current_subtitle = CurrentSubtitle()
        self._accuracy_stats = AccuracyStats()
        # 지연시간 통계 (stage 이름 → LatencyStats)
        self._latency_stats: dict = {}

    # =========================================================================
    # 오디오 레벨
    # =========================================================================

    def update_audio_level(self, rms: float, peak: float) -> None:
        """오디오 RMS/Peak 레벨을 업데이트합니다."""
        with self._lock:
            self._audio_level = AudioLevel(
                rms=rms, peak=peak, updated_at_ns=time.time_ns()
            )

    def get_audio_level(self) -> AudioLevel:
        """현재 오디오 레벨을 반환합니다."""
        with self._lock:
            return AudioLevel(
                rms=self._audio_level.rms,
                peak=self._audio_level.peak,
                updated_at_ns=self._audio_level.updated_at_ns,
            )

    # =========================================================================
    # STT 상태
    # =========================================================================

    def update_stt_status(
        self,
        connected: bool,
        error_count: Optional[int] = None,
        reconnect_count: Optional[int] = None,
    ) -> None:
        """STT 연결 상태를 업데이트합니다."""
        with self._lock:
            self._stt_status.connected = connected
            if error_count is not None:
                self._stt_status.error_count = error_count
            if reconnect_count is not None:
                self._stt_status.reconnect_count = reconnect_count
            self._stt_status.updated_at_ns = time.time_ns()

    def record_stt_result(self) -> None:
        """STT 결과 수신 시각을 기록합니다."""
        with self._lock:
            self._stt_status.last_result_at_ns = time.time_ns()

    def get_stt_status(self) -> STTStatus:
        """현재 STT 상태를 반환합니다."""
        with self._lock:
            s = self._stt_status
            return STTStatus(
                connected=s.connected,
                error_count=s.error_count,
                reconnect_count=s.reconnect_count,
                last_result_at_ns=s.last_result_at_ns,
                updated_at_ns=s.updated_at_ns,
            )

    # =========================================================================
    # 프레임 통계
    # =========================================================================

    def update_frame_stats(
        self, total_frames: int, drop_count: int, drop_rate: float, queue_depth: int
    ) -> None:
        """비디오 프레임 통계를 업데이트합니다."""
        with self._lock:
            self._frame_stats = FrameStats(
                total_frames=total_frames,
                drop_count=drop_count,
                drop_rate=drop_rate,
                queue_depth=queue_depth,
                updated_at_ns=time.time_ns(),
            )

    def get_frame_stats(self) -> FrameStats:
        """현재 프레임 통계를 반환합니다."""
        with self._lock:
            f = self._frame_stats
            return FrameStats(
                total_frames=f.total_frames,
                drop_count=f.drop_count,
                drop_rate=f.drop_rate,
                queue_depth=f.queue_depth,
                updated_at_ns=f.updated_at_ns,
            )

    # =========================================================================
    # 자막 현황
    # =========================================================================

    def update_subtitle(self, text: str, is_partial: bool) -> None:
        """현재 자막 텍스트를 업데이트합니다."""
        with self._lock:
            self._current_subtitle = CurrentSubtitle(
                text=text,
                is_partial=is_partial,
                updated_at_ns=time.time_ns(),
            )

    def get_current_subtitle(self) -> CurrentSubtitle:
        """현재 자막 정보를 반환합니다."""
        with self._lock:
            s = self._current_subtitle
            return CurrentSubtitle(
                text=s.text,
                is_partial=s.is_partial,
                updated_at_ns=s.updated_at_ns,
            )

    # =========================================================================
    # 지연시간 통계
    # =========================================================================

    def update_latency_stats(self, stage: str, stats) -> None:
        """특정 단계의 지연시간 통계를 업데이트합니다."""
        with self._lock:
            self._latency_stats[stage] = stats

    def get_latency_stats(self, stage: Optional[str] = None) -> dict:
        """지연시간 통계를 반환합니다. stage=None이면 전체를 반환합니다."""
        with self._lock:
            if stage is not None:
                return self._latency_stats.get(stage)
            return dict(self._latency_stats)

    # =========================================================================
    # 정확도 통계
    # =========================================================================

    def update_accuracy_stats(self, wer: float, cer: float, pair_count: int) -> None:
        """WER/CER 정확도 통계를 업데이트합니다."""
        with self._lock:
            self._accuracy_stats = AccuracyStats(
                wer=wer,
                cer=cer,
                pair_count=pair_count,
                updated_at_ns=time.time_ns(),
            )

    def get_accuracy_stats(self) -> AccuracyStats:
        """현재 정확도 통계를 반환합니다."""
        with self._lock:
            a = self._accuracy_stats
            return AccuracyStats(
                wer=a.wer,
                cer=a.cer,
                pair_count=a.pair_count,
                updated_at_ns=a.updated_at_ns,
            )
