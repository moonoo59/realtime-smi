"""
오디오 리샘플러 모듈입니다.

역할:
- SDI 임베디드 오디오(48kHz/24bit/stereo)를 Clova Speech STT 입력 포맷(16kHz/16bit/mono)으로 변환
- 선택 채널 추출, 스테레오→모노 믹스다운, 비트뎁스 변환, 리샘플링 수행
- 게인 조정(dB), RMS/Peak 레벨 계산
- chunk_size_ms 단위 분할
- WebRTC VAD 기반 음성 구간 필터링

사용 예시:
    >>> resampler = AudioResampler(config)
    >>> pcm_chunk = resampler.resample(audio_packet)  # AudioPacket → PCMChunk
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from src.audio import PCMChunk
from src.audio.vad_filter import VadFilter
from src.capture import AudioPacket
from src.config.schema import AppConfig

logger = logging.getLogger(__name__)

# STT 출력 포맷 고정값
_OUTPUT_SAMPLE_RATE = 16000
_OUTPUT_BIT_DEPTH = 16
_OUTPUT_CHANNELS = 1

# int16 최대값 (정규화 기준)
_INT16_MAX = 32767.0


class AudioResampler:
    """
    PCM 오디오를 리샘플링하여 STT 입력 포맷(16kHz/16bit/mono)으로 변환하는 클래스입니다.

    변환 파이프라인:
        AudioPacket(N kHz / M bit / C ch)
            → 채널 선택/믹스다운 (C → 1)
            → 비트뎁스 정규화 (M bit → float32)
            → 게인 적용 (dB)
            → scipy 리샘플링 (N kHz → 16 kHz)
            → 16bit 양자화 (float32 → int16)
            → chunk_size_ms 단위 분할
            → VAD 필터링
            → PCMChunk 목록 반환
    """

    def __init__(self, config: AppConfig) -> None:
        """
        AudioResampler를 초기화합니다.

        파라미터:
            config (AppConfig): 전체 애플리케이션 설정 객체
        """
        self._config = config
        self._audio_cfg = config.audio
        self._capture_cfg = config.capture

        # 출력 샘플링레이트
        self._out_sample_rate: int = self._audio_cfg.output_sample_rate
        # STT 전송 청크 크기 (샘플 수)
        self._chunk_samples: int = int(
            self._out_sample_rate * self._audio_cfg.chunk_size_ms / 1000
        )
        # 게인 배율 (dB → 선형)
        self._gain_linear: float = _db_to_linear(self._audio_cfg.gain_db)
        # 잔여 샘플 버퍼 (청크로 분할 후 남은 샘플)
        self._residual: Optional[np.ndarray] = None
        # 청크 순번
        self._chunk_id: int = 0
        # VAD 필터
        self._vad = VadFilter(
            mode=self._audio_cfg.vad.mode,
            sample_rate=self._out_sample_rate,
            padding_ms=self._audio_cfg.vad.padding_ms,
            enabled=self._audio_cfg.vad.enabled,
        )

        logger.info(
            f"AudioResampler 초기화: "
            f"출력={self._out_sample_rate}Hz/16bit/mono, "
            f"chunk={self._audio_cfg.chunk_size_ms}ms({self._chunk_samples}samples), "
            f"gain={self._audio_cfg.gain_db}dB"
        )

    # =========================================================================
    # 공개 메서드
    # =========================================================================

    def resample(self, packet: AudioPacket) -> list[PCMChunk]:
        """
        AudioPacket을 16kHz/16bit/mono PCMChunk 목록으로 변환합니다.

        변환 중 오류 발생 시 빈 목록을 반환하고 오류를 로깅합니다.

        파라미터:
            packet (AudioPacket): 원본 오디오 패킷

        반환값:
            list[PCMChunk]: 변환된 PCM 청크 목록 (0개 이상)
        """
        try:
            return self._process(packet)
        except Exception as exc:
            logger.error(
                f"AudioPacket {packet.packet_id} 리샘플링 실패, 패킷 스킵: {exc}",
                exc_info=True,
            )
            return []

    def apply_gain(self, data: bytes, gain_db: float) -> bytes:
        """
        PCM 데이터에 게인을 적용합니다.

        파라미터:
            data: 16bit PCM 데이터 (bytes)
            gain_db: 게인 값 (dB)

        반환값:
            bytes: 게인 적용된 PCM 데이터
        """
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        gain_linear = _db_to_linear(gain_db)
        samples = np.clip(samples * gain_linear, -_INT16_MAX, _INT16_MAX)
        return samples.astype(np.int16).tobytes()

    def calculate_rms(self, data: bytes) -> float:
        """
        16bit PCM 데이터의 RMS 레벨을 계산합니다.

        파라미터:
            data: 16bit PCM 데이터 (bytes)

        반환값:
            float: RMS 레벨 (0.0~1.0, 1.0 = 풀스케일)
        """
        if len(data) == 0:
            return 0.0
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) / _INT16_MAX
        return min(1.0, rms)

    def calculate_peak(self, data: bytes) -> float:
        """
        16bit PCM 데이터의 Peak 레벨을 계산합니다.

        파라미터:
            data: 16bit PCM 데이터 (bytes)

        반환값:
            float: Peak 레벨 (0.0~1.0, 1.0 = 풀스케일)
        """
        if len(data) == 0:
            return 0.0
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        peak = float(np.max(np.abs(samples))) / _INT16_MAX
        return min(1.0, peak)

    def split_chunks(self, pcm_data: bytes, chunk_size: int) -> list[bytes]:
        """
        PCM 데이터를 chunk_size 바이트 단위로 분할합니다.

        파라미터:
            pcm_data: 분할할 16bit PCM 데이터 (bytes)
            chunk_size: 청크당 바이트 수

        반환값:
            list[bytes]: 분할된 PCM 청크 목록
        """
        chunks = []
        for offset in range(0, len(pcm_data), chunk_size):
            chunks.append(pcm_data[offset:offset + chunk_size])
        return chunks

    def reset(self) -> None:
        """잔여 샘플 버퍼 및 VAD 상태를 초기화합니다."""
        self._residual = None
        self._vad.reset()
        logger.debug("AudioResampler 상태 초기화")

    def update_config(self, config) -> None:
        """
        VAD 및 게인 설정을 핫스왑으로 업데이트합니다.

        업데이트 가능 항목:
        - audio.gain_db (게인)
        - audio.vad.enabled, audio.vad.mode, audio.vad.padding_ms (VAD)

        파라미터:
            config: 새 AppConfig 객체
        """
        self._audio_cfg = config.audio
        self._gain_linear = _db_to_linear(config.audio.gain_db)
        self._vad = VadFilter(
            mode=config.audio.vad.mode,
            sample_rate=self._out_sample_rate,
            padding_ms=config.audio.vad.padding_ms,
            enabled=config.audio.vad.enabled,
        )
        logger.info(
            f"AudioResampler 설정 핫스왑: "
            f"gain_db={config.audio.gain_db}, "
            f"vad_enabled={config.audio.vad.enabled}"
        )

    # =========================================================================
    # 내부 처리 메서드
    # =========================================================================

    def _process(self, packet: AudioPacket) -> list[PCMChunk]:
        """
        AudioPacket을 처리하여 PCMChunk 목록을 생성하는 핵심 로직입니다.

        처리 단계:
        1. bytes → numpy int16 배열 변환
        2. 비트뎁스 정규화 (float32)
        3. 채널 선택/믹스다운 (→ mono)
        4. 게인 적용
        5. 리샘플링 (N kHz → 16 kHz)
        6. int16 양자화
        7. 잔여 버퍼와 병합
        8. chunk_size_ms 단위 분할
        9. VAD 필터링
        10. PCMChunk 생성
        """
        # 1. bytes → numpy 배열
        raw = _bytes_to_numpy(packet.data, packet.bit_depth, packet.channels)

        # 2. float32로 정규화 (-1.0 ~ +1.0)
        normalized = _normalize_to_float32(raw, packet.bit_depth)

        # 3. 채널 믹스다운 (→ mono)
        mono = _mixdown_to_mono(normalized, packet.channels)

        # 4. 게인 적용
        if self._gain_linear != 1.0:
            mono = np.clip(mono * self._gain_linear, -1.0, 1.0)

        # 5. 리샘플링 (입력 샘플레이트 → 16kHz)
        if packet.sample_rate != self._out_sample_rate:
            mono = _resample(mono, packet.sample_rate, self._out_sample_rate)

        # 6. int16 양자화
        pcm_int16 = np.clip(mono * _INT16_MAX, -_INT16_MAX, _INT16_MAX).astype(np.int16)

        # 7. 잔여 버퍼와 병합
        if self._residual is not None and len(self._residual) > 0:
            pcm_int16 = np.concatenate([self._residual, pcm_int16])

        # 8. chunk_size 단위 분할
        chunks: list[PCMChunk] = []
        offset = 0

        while offset + self._chunk_samples <= len(pcm_int16):
            chunk_samples = pcm_int16[offset:offset + self._chunk_samples]
            chunk_bytes = chunk_samples.tobytes()

            rms = self.calculate_rms(chunk_bytes)
            peak = self.calculate_peak(chunk_bytes)

            # 9. VAD 필터링
            if not self._vad.is_speech(chunk_bytes):
                offset += self._chunk_samples
                continue

            # 10. PCMChunk 생성
            pcm_chunk = PCMChunk(
                chunk_id=self._chunk_id,
                capture_timestamp_ns=packet.timestamp_ns,
                sample_rate=_OUTPUT_SAMPLE_RATE,
                bit_depth=_OUTPUT_BIT_DEPTH,
                channels=_OUTPUT_CHANNELS,
                data=chunk_bytes,
                rms=rms,
                peak=peak,
            )
            chunks.append(pcm_chunk)
            self._chunk_id += 1
            offset += self._chunk_samples

        # 잔여 샘플 저장 (다음 패킷 처리 시 사용)
        self._residual = pcm_int16[offset:] if offset < len(pcm_int16) else None

        if chunks:
            logger.debug(
                f"패킷 {packet.packet_id}: "
                f"{len(chunks)}개 청크 생성, "
                f"RMS={chunks[-1].rms:.3f}, Peak={chunks[-1].peak:.3f}"
            )

        return chunks


# =============================================================================
# 모듈 레벨 헬퍼 함수
# =============================================================================

def _bytes_to_numpy(data: bytes, bit_depth: int, channels: int) -> np.ndarray:
    """
    PCM bytes를 numpy 배열로 변환합니다.

    파라미터:
        data: 원시 PCM 바이트
        bit_depth: 비트뎁스 (16 또는 24)
        channels: 채널 수

    반환값:
        np.ndarray: shape=(samples, channels) 또는 (samples,) for mono
    """
    if bit_depth == 16:
        arr = np.frombuffer(data, dtype=np.int16)
    elif bit_depth == 24:
        arr = _decode_24bit(data)
    elif bit_depth == 32:
        arr = np.frombuffer(data, dtype=np.int32)
    else:
        raise ValueError(f"지원하지 않는 비트뎁스: {bit_depth}")

    if channels > 1:
        # interleaved → (samples, channels)
        arr = arr.reshape(-1, channels)

    return arr


def _decode_24bit(data: bytes) -> np.ndarray:
    """
    24bit PCM 데이터를 int32 numpy 배열로 디코딩합니다.

    24bit는 numpy 기본 타입이 없으므로 3바이트씩 읽어 int32로 변환합니다.

    파라미터:
        data: 24bit PCM 바이트 (little-endian 가정)

    반환값:
        np.ndarray: int32 배열 (24bit 부호 확장)
    """
    byte_array = np.frombuffer(data, dtype=np.uint8)
    # 3바이트 → 4바이트로 패딩 (little-endian)
    n_samples = len(byte_array) // 3
    padded = np.zeros(n_samples * 4, dtype=np.uint8)
    padded[0::4] = byte_array[0::3]
    padded[1::4] = byte_array[1::3]
    padded[2::4] = byte_array[2::3]
    # 부호 확장: bit 23이 1이면 상위 바이트를 0xFF로
    padded[3::4] = np.where((byte_array[2::3] & 0x80) != 0, 0xFF, 0x00)
    return padded.view(np.int32)


def _normalize_to_float32(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    정수 PCM 배열을 -1.0~+1.0 float32로 정규화합니다.

    파라미터:
        arr: 정수형 numpy 배열
        bit_depth: 원본 비트뎁스

    반환값:
        np.ndarray: float32 배열 (-1.0~+1.0)
    """
    max_val = float(2 ** (bit_depth - 1))
    return arr.astype(np.float32) / max_val


def _mixdown_to_mono(arr: np.ndarray, channels: int) -> np.ndarray:
    """
    멀티채널 배열을 모노로 믹스다운합니다.

    스테레오 이상의 경우 모든 채널의 평균을 취합니다.

    파라미터:
        arr: shape=(samples, channels) 또는 (samples,) float32 배열
        channels: 채널 수

    반환값:
        np.ndarray: shape=(samples,) float32 mono 배열
    """
    if channels == 1 or arr.ndim == 1:
        return arr.flatten()
    # 채널 평균
    return arr.mean(axis=1)


def _resample(mono: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    scipy.signal.resample_poly를 사용하여 샘플레이트를 변환합니다.

    up/down 비율을 GCD로 약분하여 연산량을 최소화합니다.

    파라미터:
        mono: float32 mono 배열
        from_rate: 입력 샘플레이트 (Hz)
        to_rate: 출력 샘플레이트 (Hz)

    반환값:
        np.ndarray: 리샘플링된 float32 배열
    """
    from math import gcd
    common = gcd(from_rate, to_rate)
    up = to_rate // common
    down = from_rate // common
    return resample_poly(mono, up, down).astype(np.float32)


def _db_to_linear(db: float) -> float:
    """dB 값을 선형 배율로 변환합니다."""
    return 10.0 ** (db / 20.0)
