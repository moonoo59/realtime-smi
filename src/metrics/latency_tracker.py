"""
파이프라인 단계별 지연시간 추적 모듈입니다.

역할:
- 캡처, STT 전송, STT 수신, 화면 표시 시각을 nanoseconds 단위로 기록
- 60초 슬라이딩 윈도우 기반 P95/P99 통계 계산
- End-to-end 지연시간 조회
- CSV 파일 내보내기

사용 예시:
    >>> tracker = LatencyTracker(config)
    >>> tracker.record_capture(packet_id=0, timestamp_ns=time.time_ns())
    >>> tracker.record_stt_send(packet_id=0, timestamp_ns=time.time_ns())
    >>> stats = tracker.compute_stats(window_sec=60)
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.schema import AppConfig
from src.metrics import LatencyStats

logger = logging.getLogger(__name__)


@dataclass
class _TimestampRecord:
    """단일 오디오/비디오 패킷의 타임스탬프 기록입니다."""
    packet_id: int
    capture_ns: Optional[int] = None
    stt_send_ns: Optional[int] = None
    stt_receive_ns: Optional[int] = None
    display_ns: Optional[int] = None
    result_type: Optional[str] = None  # "partial" | "final"

    def capture_to_send_ms(self) -> Optional[float]:
        """캡처 → STT 전송 지연 (ms)"""
        if self.capture_ns is not None and self.stt_send_ns is not None:
            return (self.stt_send_ns - self.capture_ns) / 1_000_000
        return None

    def send_to_receive_ms(self) -> Optional[float]:
        """STT 전송 → 수신 지연 (ms)"""
        if self.stt_send_ns is not None and self.stt_receive_ns is not None:
            return (self.stt_receive_ns - self.stt_send_ns) / 1_000_000
        return None

    def end_to_end_ms(self) -> Optional[float]:
        """캡처 → 화면 표시 end-to-end 지연 (ms)"""
        if self.capture_ns is not None and self.display_ns is not None:
            return (self.display_ns - self.capture_ns) / 1_000_000
        return None


class LatencyTracker:
    """
    파이프라인 단계별 지연시간을 추적하고 통계를 계산하는 클래스입니다.

    각 패킷/결과에 대해 4개 시각을 기록합니다:
    1. capture_ns: AudioPacket 캡처 시각
    2. stt_send_ns: STT 스트리머에 전송한 시각
    3. stt_receive_ns: STT 결과 수신 시각
    4. display_ns: 화면에 표시된 시각

    통계 계산:
    - 설정된 latency_window_sec 이내의 레코드만 사용
    - numpy.percentile로 P95/P99 계산
    """

    # 추적할 단계 이름
    STAGE_CAPTURE_TO_SEND = "capture_to_stt_send"
    STAGE_SEND_TO_PARTIAL = "stt_send_to_partial"
    STAGE_SEND_TO_FINAL = "stt_send_to_final"
    STAGE_END_TO_END = "end_to_end"

    def __init__(self, config: AppConfig) -> None:
        self._window_sec = config.metrics.latency_window_sec
        self._output_dir = Path(config.metrics.metrics_output_dir)
        self._lock = threading.RLock()

        # packet_id → 타임스탬프 레코드
        self._records: dict[int, _TimestampRecord] = {}
        # result_id → packet_id 매핑 (STT 결과와 오디오 패킷 연결)
        self._result_to_packet: dict[int, int] = {}
        # result_id → 타임스탬프 레코드 (STT 결과 기준)
        self._result_records: dict[int, _TimestampRecord] = {}

        # 완성된 측정값 버퍼 (슬라이딩 윈도우)
        # stage → [(timestamp_ns, latency_ms), ...]
        self._measurements: dict[str, list[tuple[int, float]]] = defaultdict(list)

        logger.info(
            f"LatencyTracker 초기화: "
            f"window={self._window_sec}초, "
            f"output={self._output_dir}"
        )

    # =========================================================================
    # 타임스탬프 기록 메서드
    # =========================================================================

    def record_capture(self, packet_id: int, timestamp_ns: int) -> None:
        """
        오디오 패킷 캡처 시각을 기록합니다.

        파라미터:
            packet_id: 패킷 순번
            timestamp_ns: 캡처 시각 (nanoseconds)
        """
        with self._lock:
            if packet_id not in self._records:
                self._records[packet_id] = _TimestampRecord(packet_id=packet_id)
            self._records[packet_id].capture_ns = timestamp_ns

    def record_stt_send(self, packet_id: int, timestamp_ns: int) -> None:
        """
        오디오 청크 STT 전송 시각을 기록하고, capture→send 지연을 측정합니다.

        파라미터:
            packet_id: 패킷 순번 (PCMChunk의 chunk_id와 매핑)
            timestamp_ns: 전송 시각 (nanoseconds)
        """
        with self._lock:
            if packet_id not in self._records:
                self._records[packet_id] = _TimestampRecord(packet_id=packet_id)
            self._records[packet_id].stt_send_ns = timestamp_ns

            # capture → send 지연 측정
            latency_ms = self._records[packet_id].capture_to_send_ms()
            if latency_ms is not None:
                self._add_measurement(
                    self.STAGE_CAPTURE_TO_SEND, timestamp_ns, latency_ms
                )

    def record_stt_receive(
        self,
        result_id: int,
        packet_id: int,
        timestamp_ns: int,
        result_type: str,
    ) -> None:
        """
        STT 결과 수신 시각을 기록하고, send→receive 지연을 측정합니다.

        파라미터:
            result_id: STTResult 순번
            packet_id: 연관된 오디오 패킷 순번
            timestamp_ns: 수신 시각 (nanoseconds)
            result_type: "partial" | "final"
        """
        with self._lock:
            self._result_to_packet[result_id] = packet_id

            if packet_id in self._records:
                record = self._records[packet_id]
                record.stt_receive_ns = timestamp_ns
                record.result_type = result_type

                latency_ms = record.send_to_receive_ms()
                if latency_ms is not None:
                    stage = (
                        self.STAGE_SEND_TO_PARTIAL
                        if result_type == "partial"
                        else self.STAGE_SEND_TO_FINAL
                    )
                    self._add_measurement(stage, timestamp_ns, latency_ms)

    def record_display(self, result_id: int, timestamp_ns: int) -> None:
        """
        자막 화면 표시 시각을 기록하고, end-to-end 지연을 측정합니다.

        파라미터:
            result_id: STTResult 순번
            timestamp_ns: 표시 시각 (nanoseconds)
        """
        with self._lock:
            packet_id = self._result_to_packet.get(result_id)
            if packet_id is None or packet_id not in self._records:
                return

            record = self._records[packet_id]
            record.display_ns = timestamp_ns

            e2e_ms = record.end_to_end_ms()
            if e2e_ms is not None:
                self._add_measurement(self.STAGE_END_TO_END, timestamp_ns, e2e_ms)

    # =========================================================================
    # 통계 계산
    # =========================================================================

    def compute_stats(self, window_sec: Optional[int] = None) -> dict[str, LatencyStats]:
        """
        슬라이딩 윈도우 내 지연시간 통계를 계산합니다.

        파라미터:
            window_sec: 윈도우 크기 (초). None이면 config 기본값 사용

        반환값:
            dict[str, LatencyStats]: 단계별 통계
        """
        window = window_sec or self._window_sec
        cutoff_ns = time.time_ns() - int(window * 1_000_000_000)

        stats = {}
        with self._lock:
            for stage, measurements in self._measurements.items():
                # 윈도우 내 측정값만 필터링
                recent = [ms for ts, ms in measurements if ts >= cutoff_ns]
                if not recent:
                    continue

                arr = np.array(recent, dtype=np.float64)
                stats[stage] = LatencyStats(
                    stage=stage,
                    count=len(arr),
                    mean_ms=float(np.mean(arr)),
                    min_ms=float(np.min(arr)),
                    max_ms=float(np.max(arr)),
                    p95_ms=float(np.percentile(arr, 95)),
                    p99_ms=float(np.percentile(arr, 99)),
                )

        if stats:
            logger.info(
                f"지연시간 통계 ({window}초 윈도우): "
                + ", ".join(
                    f"{s}=avg{v.mean_ms:.0f}ms/P95={v.p95_ms:.0f}ms"
                    for s, v in stats.items()
                )
            )

        return stats

    def get_e2e_latency(self, result_id: int) -> Optional[float]:
        """
        단일 결과의 end-to-end 지연시간을 반환합니다.

        파라미터:
            result_id: STTResult 순번

        반환값:
            Optional[float]: end-to-end 지연 (ms). 데이터 부족 시 None
        """
        with self._lock:
            packet_id = self._result_to_packet.get(result_id)
            if packet_id is None:
                return None
            record = self._records.get(packet_id)
            if record is None:
                return None
            return record.end_to_end_ms()

    def export_csv(self, filepath: str | Path) -> None:
        """
        전체 타임스탬프 기록을 CSV 파일로 저장합니다.

        CSV 컬럼:
            packet_id, capture_ns, stt_send_ns, stt_receive_ns, display_ns,
            result_type, capture_to_send_ms, send_to_receive_ms, end_to_end_ms

        파라미터:
            filepath: 저장할 CSV 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            records = list(self._records.values())

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "packet_id", "capture_ns", "stt_send_ns", "stt_receive_ns",
                    "display_ns", "result_type",
                    "capture_to_send_ms", "send_to_receive_ms", "end_to_end_ms",
                ])
                for r in records:
                    writer.writerow([
                        r.packet_id,
                        r.capture_ns or "",
                        r.stt_send_ns or "",
                        r.stt_receive_ns or "",
                        r.display_ns or "",
                        r.result_type or "",
                        f"{r.capture_to_send_ms():.3f}" if r.capture_to_send_ms() is not None else "",
                        f"{r.send_to_receive_ms():.3f}" if r.send_to_receive_ms() is not None else "",
                        f"{r.end_to_end_ms():.3f}" if r.end_to_end_ms() is not None else "",
                    ])

            logger.info(f"지연시간 CSV 저장 완료: {filepath} ({len(records)}개 레코드)")

        except OSError as exc:
            logger.error(f"지연시간 CSV 저장 실패: {exc}")
            raise

    # =========================================================================
    # 내부 헬퍼
    # =========================================================================

    def _add_measurement(self, stage: str, timestamp_ns: int, latency_ms: float) -> None:
        """측정값을 버퍼에 추가하고, 오래된 데이터를 정리합니다."""
        self._measurements[stage].append((timestamp_ns, latency_ms))

        # 윈도우 2배 이상 오래된 데이터 정리 (메모리 절약)
        cutoff_ns = time.time_ns() - int(self._window_sec * 2 * 1_000_000_000)
        self._measurements[stage] = [
            (ts, ms) for ts, ms in self._measurements[stage] if ts >= cutoff_ns
        ]
