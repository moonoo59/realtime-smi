"""
캡처 모듈 패키지

공통 데이터 타입 정의:
- VideoFrame: 비디오 프레임 컨테이너
- AudioPacket: 오디오 패킷 컨테이너
"""

from dataclasses import dataclass


@dataclass
class VideoFrame:
    """
    비디오 프레임 데이터 컨테이너입니다.

    필드:
        frame_id: 프레임 순번 (0부터 시작)
        timestamp_ns: 캡처 시각 (nanoseconds, time.time_ns() 기준)
        width: 프레임 가로 픽셀 수
        height: 프레임 세로 픽셀 수
        pixel_format: 픽셀 포맷 문자열 ("yuv422" | "bgra")
        data: 원시 픽셀 데이터 (bytes)
    """
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    pixel_format: str
    data: bytes


@dataclass
class AudioPacket:
    """
    오디오 패킷 데이터 컨테이너입니다.

    필드:
        packet_id: 패킷 순번 (0부터 시작)
        timestamp_ns: 캡처 시각 (nanoseconds, time.time_ns() 기준)
        sample_rate: 샘플링레이트 (Hz, 예: 48000)
        bit_depth: 비트뎁스 (예: 16, 24)
        channels: 채널 수 (예: 1=mono, 2=stereo)
        data: 원시 PCM 데이터 (bytes)
    """
    packet_id: int
    timestamp_ns: int
    sample_rate: int
    bit_depth: int
    channels: int
    data: bytes
