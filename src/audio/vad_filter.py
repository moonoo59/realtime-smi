"""
WebRTC VAD(Voice Activity Detection) 필터 모듈입니다.

역할:
- WebRTC VAD 라이브러리를 래핑하여 음성/비음성 구간을 판별
- 음성 종료 후 padding_ms 동안 추가 오디오를 전송하여 단어 끝 누락 방지
- VAD가 비활성화된 경우 항상 True를 반환하여 모든 청크를 통과

사용 예시:
    >>> vad = VadFilter(mode=3, sample_rate=16000, padding_ms=300)
    >>> is_speech = vad.is_speech(pcm_chunk_bytes)  # 16kHz/16bit/mono 필수
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# webrtcvad 설치 여부 확인 (선택적 의존성)
try:
    import webrtcvad as _webrtcvad
    _VAD_AVAILABLE = True
except ImportError:
    _VAD_AVAILABLE = False
    logger.warning(
        "webrtcvad 라이브러리가 설치되지 않았습니다. "
        "VAD 기능이 비활성화됩니다. "
        "'pip install webrtcvad'로 설치하세요."
    )


class VadFilter:
    """
    WebRTC VAD 기반 음성 구간 판별 필터입니다.

    WebRTC VAD는 10ms, 20ms, 30ms 단위 프레임만 지원합니다.
    청크가 이 크기가 아닌 경우 내부에서 분할하여 처리합니다.

    음성 종료 후 padding_ms 동안은 계속 음성으로 판별합니다.
    이를 통해 문장 끝 단어가 잘리는 것을 방지합니다.

    파라미터:
        mode: VAD 공격성 (0~3, 3=가장 공격적으로 비음성을 걸러냄)
        sample_rate: 입력 샘플링레이트 (Hz, 16000 고정)
        padding_ms: 음성 종료 후 추가 패딩 시간 (밀리초)
        enabled: VAD 활성화 여부 (False이면 항상 True 반환)
    """

    # WebRTC VAD가 지원하는 프레임 길이 (ms)
    _VALID_FRAME_MS = (10, 20, 30)

    def __init__(
        self,
        mode: int = 3,
        sample_rate: int = 16000,
        padding_ms: int = 300,
        enabled: bool = True,
    ) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._padding_ms = padding_ms
        self._enabled = enabled and _VAD_AVAILABLE

        # 패딩 카운터 (음성 종료 후 남은 패딩 ms)
        self._padding_remaining_ms: int = 0

        # WebRTC VAD 인스턴스 생성
        self._vad = None
        if self._enabled:
            self._vad = _webrtcvad.Vad(mode)
            logger.info(
                f"VadFilter 초기화: mode={mode}, "
                f"sample_rate={sample_rate}Hz, "
                f"padding={padding_ms}ms"
            )
        else:
            logger.info("VadFilter 비활성화 (모든 청크를 음성으로 처리)")

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        PCM 청크가 음성 구간인지 판별합니다.

        VAD가 비활성화된 경우 항상 True를 반환합니다.
        패딩 기간 중에는 VAD 결과와 무관하게 True를 반환합니다.

        파라미터:
            pcm_bytes: 16kHz/16bit/mono PCM 데이터 (bytes)

        반환값:
            bool: 음성 구간이면 True, 비음성 구간이면 False
        """
        if not self._enabled:
            return True

        # 청크를 WebRTC VAD 지원 프레임 크기로 분할하여 분석
        frame_ms = 30  # 30ms 프레임 사용
        speech_detected = self._analyze_frames(pcm_bytes, frame_ms)

        if speech_detected:
            # 음성 감지 시 패딩 카운터 리셋
            self._padding_remaining_ms = self._padding_ms
            return True
        elif self._padding_remaining_ms > 0:
            # 패딩 기간 중 - 음성으로 처리하고 카운터 감소
            # 청크 길이에 해당하는 ms를 패딩에서 차감
            chunk_ms = len(pcm_bytes) / (self._sample_rate * 2) * 1000
            self._padding_remaining_ms = max(
                0, self._padding_remaining_ms - int(chunk_ms)
            )
            return True
        else:
            return False

    def reset(self) -> None:
        """패딩 상태를 초기화합니다."""
        self._padding_remaining_ms = 0

    # =========================================================================
    # 내부 메서드
    # =========================================================================

    def _analyze_frames(self, pcm_bytes: bytes, frame_ms: int) -> bool:
        """
        PCM 데이터를 frame_ms 단위로 분할하여 VAD를 분석합니다.

        하나라도 음성 프레임이 있으면 True를 반환합니다.

        파라미터:
            pcm_bytes: 분석할 PCM 데이터
            frame_ms: 프레임 길이 (10, 20, 30ms 중 하나)

        반환값:
            bool: 음성 프레임이 하나라도 있으면 True
        """
        # 프레임당 바이트 수 계산 (16bit = 2바이트/샘플)
        bytes_per_frame = int(self._sample_rate * frame_ms / 1000) * 2

        offset = 0
        while offset + bytes_per_frame <= len(pcm_bytes):
            frame = pcm_bytes[offset:offset + bytes_per_frame]
            try:
                if self._vad.is_speech(frame, self._sample_rate):
                    return True
            except Exception:
                # VAD 오류 시 음성으로 처리 (안전한 기본값)
                return True
            offset += bytes_per_frame

        return False
