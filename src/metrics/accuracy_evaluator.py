"""
WER/CER 정확도 평가 모듈입니다.

역할:
- jiwer를 사용한 WER(Word Error Rate) 계산
- 문자 단위 CER(Character Error Rate) 계산
- AccuracyReport 생성 및 JSON/TXT 파일 저장
- 레퍼런스 텍스트 파일 로드

사용 예시:
    >>> evaluator = AccuracyEvaluator(config)
    >>> evaluator.add_result(hypothesis="안녕하세요", reference="안녕하세요")
    >>> report = evaluator.compute_report(session_id="session-1")
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.config.schema import AppConfig
from src.metrics import AccuracyReport, ErrorDetail

logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    """
    WER/CER 정확도를 계산하고 리포트를 생성하는 클래스입니다.

    jiwer 라이브러리를 사용하여 단어/문자 오류율을 계산합니다.
    레퍼런스 텍스트는 파일에서 순서대로 읽거나, 직접 전달할 수 있습니다.
    """

    def __init__(self, config: AppConfig) -> None:
        self._enabled = config.accuracy.enabled
        self._reference_source = config.accuracy.reference_source
        self._reference_file = config.accuracy.reference_file
        self._output_dir = Path(config.accuracy.output_dir)
        self._report_interval_sec = config.accuracy.report_interval_sec

        self._lock = threading.RLock()
        # (hypothesis, reference) 쌍 목록
        self._pairs: list[tuple[str, str]] = []
        self._session_start_ns: int = time.time_ns()

        # 레퍼런스 파일 라인 목록 (reference_source=file일 때)
        self._reference_lines: list[str] = []
        self._reference_index: int = 0

        if self._reference_source == "file" and self._reference_file:
            self._load_reference_file(self._reference_file)

        logger.info(
            f"AccuracyEvaluator 초기화: "
            f"enabled={self._enabled}, "
            f"source={self._reference_source}, "
            f"output={self._output_dir}"
        )

    # =========================================================================
    # 레퍼런스 파일 로드
    # =========================================================================

    def _load_reference_file(self, filepath: str) -> None:
        """레퍼런스 텍스트 파일을 라인 단위로 로드합니다."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"레퍼런스 파일 없음: {filepath}")
            return
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._reference_lines = lines
        logger.info(f"레퍼런스 파일 로드: {filepath} ({len(lines)}개 라인)")

    # =========================================================================
    # 결과 추가
    # =========================================================================

    def add_result(self, hypothesis: str, reference: Optional[str] = None) -> None:
        """
        STT 인식 결과와 레퍼런스를 추가합니다.

        파라미터:
            hypothesis: STT 인식 텍스트
            reference: 정답 텍스트. None이면 파일에서 순서대로 읽음
        """
        if not self._enabled:
            return

        with self._lock:
            if reference is None:
                if self._reference_index < len(self._reference_lines):
                    reference = self._reference_lines[self._reference_index]
                    self._reference_index += 1
                else:
                    logger.debug("레퍼런스 소진: 추가 건너뜀")
                    return

            self._pairs.append((hypothesis, reference))

    def add_result_with_reference(self, hypothesis: str, reference: str) -> None:
        """레퍼런스를 직접 지정하여 결과를 추가합니다."""
        self.add_result(hypothesis=hypothesis, reference=reference)

    # =========================================================================
    # 정확도 계산
    # =========================================================================

    def compute_wer(self, hypothesis: str, reference: str) -> float:
        """
        단일 쌍에 대한 WER을 계산합니다.

        파라미터:
            hypothesis: 인식 텍스트
            reference: 정답 텍스트

        반환값:
            float: WER (0.0~1.0+, 1.0 초과 가능)
        """
        try:
            import jiwer
            return float(jiwer.wer(reference=reference, hypothesis=hypothesis))
        except Exception as exc:
            logger.error(f"WER 계산 실패: {exc}")
            return 1.0

    def compute_cer(self, hypothesis: str, reference: str) -> float:
        """
        단일 쌍에 대한 CER을 계산합니다.

        파라미터:
            hypothesis: 인식 텍스트
            reference: 정답 텍스트

        반환값:
            float: CER (0.0~1.0+)
        """
        try:
            import jiwer
            return float(jiwer.cer(reference=reference, hypothesis=hypothesis))
        except Exception as exc:
            logger.error(f"CER 계산 실패: {exc}")
            return 1.0

    def compute_report(self, session_id: str = "") -> AccuracyReport:
        """
        누적된 모든 결과에 대한 정확도 리포트를 계산합니다.

        파라미터:
            session_id: 세션 식별자

        반환값:
            AccuracyReport: 집계된 WER/CER 리포트
        """
        import jiwer

        with self._lock:
            pairs = list(self._pairs)

        duration_sec = (time.time_ns() - self._session_start_ns) / 1_000_000_000

        if not pairs:
            return AccuracyReport(
                session_id=session_id,
                duration_sec=duration_sec,
                total_words=0,
                total_chars=0,
                wer=0.0,
                cer=0.0,
                error_details=[],
            )

        hypotheses = [h for h, _ in pairs]
        references = [r for _, r in pairs]

        # 전체 WER/CER 계산 (jiwer는 리스트 입력 지원)
        try:
            overall_wer = jiwer.wer(reference=references, hypothesis=hypotheses)
        except Exception as exc:
            logger.error(f"전체 WER 계산 실패: {exc}")
            overall_wer = 0.0

        try:
            overall_cer = jiwer.cer(reference=references, hypothesis=hypotheses)
        except Exception as exc:
            logger.error(f"전체 CER 계산 실패: {exc}")
            overall_cer = 0.0

        # 총 단어 수 / 문자 수
        total_words = sum(len(r.split()) for r in references)
        total_chars = sum(len(r.replace(" ", "")) for r in references)

        # 개별 오류 상세 (WER > 0인 쌍만)
        error_details: list[ErrorDetail] = []
        for hyp, ref in pairs:
            try:
                pair_wer = jiwer.wer(reference=ref, hypothesis=hyp)
            except Exception:
                pair_wer = 1.0
            if pair_wer > 0:
                error_details.append(ErrorDetail(
                    hypothesis=hyp,
                    reference=ref,
                    wer=pair_wer,
                ))

        report = AccuracyReport(
            session_id=session_id,
            duration_sec=duration_sec,
            total_words=total_words,
            total_chars=total_chars,
            wer=overall_wer,
            cer=overall_cer,
            error_details=error_details,
        )

        logger.info(
            f"정확도 리포트: session={session_id}, "
            f"pairs={len(pairs)}, WER={overall_wer:.3f}, CER={overall_cer:.3f}"
        )

        return report

    # =========================================================================
    # 리포트 저장
    # =========================================================================

    def save_report(self, report: AccuracyReport, filepath: Optional[str | Path] = None) -> Path:
        """
        AccuracyReport를 JSON 파일로 저장합니다.

        파라미터:
            report: 저장할 리포트
            filepath: 저장 경로. None이면 output_dir에 자동 생성

        반환값:
            Path: 저장된 파일 경로
        """
        if filepath is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            filename = f"accuracy_{report.session_id or ts}.json"
            filepath = self._output_dir / filename
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": report.session_id,
            "duration_sec": report.duration_sec,
            "total_words": report.total_words,
            "total_chars": report.total_chars,
            "wer": report.wer,
            "cer": report.cer,
            "error_count": len(report.error_details),
            "error_details": [
                {"hypothesis": e.hypothesis, "reference": e.reference, "wer": e.wer}
                for e in report.error_details
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"정확도 리포트 저장: {filepath}")
        return filepath

    # =========================================================================
    # 상태 초기화
    # =========================================================================

    def reset(self) -> None:
        """누적된 결과를 초기화합니다."""
        with self._lock:
            self._pairs.clear()
            self._reference_index = 0
            self._session_start_ns = time.time_ns()
        logger.info("AccuracyEvaluator 초기화 완료")

    def get_pair_count(self) -> int:
        """누적된 (hypothesis, reference) 쌍의 수를 반환합니다."""
        with self._lock:
            return len(self._pairs)
