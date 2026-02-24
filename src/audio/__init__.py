"""
오디오 처리 모듈 패키지

공통 데이터 타입:
- PCMChunk: STT 입력용 16kHz/16bit/mono PCM 청크 컨테이너
"""

from dataclasses import dataclass


@dataclass
class PCMChunk:
    """
    STT 입력용 PCM 오디오 청크 컨테이너입니다.

    AudioResampler가 AudioPacket을 변환하여 생성합니다.

    필드:
        chunk_id: 청크 순번 (0부터 시작)
        capture_timestamp_ns: 원본 캡처 시각 (AudioPacket.timestamp_ns 기준)
        sample_rate: 샘플링레이트 (Hz, 항상 16000)
        bit_depth: 비트뎁스 (항상 16)
        channels: 채널 수 (항상 1, mono)
        data: 16kHz/16bit/mono PCM 바이트 데이터
        rms: RMS 레벨 (0.0~1.0)
        peak: Peak 레벨 (0.0~1.0)
    """
    chunk_id: int
    capture_timestamp_ns: int
    sample_rate: int   # 16000
    bit_depth: int     # 16
    channels: int      # 1
    data: bytes
    rms: float
    peak: float
