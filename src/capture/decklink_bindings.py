"""
Blackmagic DeckLink SDK ctypes 바인딩 모듈입니다.

역할:
- DeckLink SDK C API를 Python ctypes로 래핑
- IDeckLinkIterator, IDeckLink, IDeckLinkInput,
  IDeckLinkVideoInputFrame, IDeckLinkAudioInputPacket 인터페이스 바인딩
- IDeckLinkInputCallback Python 구현체 (역방향 vtable 구성)
- macOS/Linux 플랫폼별 라이브러리 경로 처리

SDK 설치 경로:
    macOS: /Library/Frameworks/DeckLinkAPI.framework/DeckLinkAPI
    Linux: /usr/lib/libDeckLinkAPI.so  (또는 ldconfig 탐색)

vtable 슬롯 인덱스 기준: Blackmagic Desktop Video SDK 12.x
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import platform
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# 예외 클래스
# =============================================================================

class DeckLinkSDKNotFoundError(RuntimeError):
    """DeckLink SDK 라이브러리를 찾을 수 없을 때 발생합니다."""


class DeckLinkDeviceNotFoundError(RuntimeError):
    """지정한 인덱스의 DeckLink 장치가 없을 때 발생합니다."""


class DeckLinkAPIError(RuntimeError):
    """DeckLink API 호출이 실패했을 때 발생합니다."""


# =============================================================================
# HRESULT 상수
# =============================================================================

S_OK    = 0
S_FALSE = 1


# =============================================================================
# BMD 상수
# =============================================================================

class BMDDisplayMode:
    """BMDDisplayMode 상수 (DeckLink SDK 12.x 기준)"""
    HD1080p30 = 0x48703330  # 'Hp30'
    HD1080p25 = 0x48703235  # 'Hp25'
    HD1080i50 = 0x48693530  # 'Hi50'
    HD720p60  = 0x68703630  # 'hp60'


class BMDPixelFormat:
    """BMDPixelFormat 상수"""
    YUV422_8bit  = 0x32767579  # '2vuy' (UYVY, 8bit)
    YUV422_10bit = 0x76323130  # 'v210' (10bit)
    BGRA_8bit    = 0x42475241  # 'BGRA'


class BMDAudioSampleRate:
    """BMDAudioSampleRate 상수"""
    Rate48kHz = 48000


class BMDAudioSampleType:
    """BMDAudioSampleType 상수"""
    Int16bit = 16
    Int32bit = 32


class BMDVideoInputFlags:
    """BMDVideoInputFlags 상수"""
    Default = 0


# 설정 문자열 → BMDDisplayMode 매핑
VIDEO_MODE_MAP: dict[str, int] = {
    '1080p30': BMDDisplayMode.HD1080p30,
    '1080p25': BMDDisplayMode.HD1080p25,
    '1080i50': BMDDisplayMode.HD1080i50,
    '720p60':  BMDDisplayMode.HD720p60,
}


# =============================================================================
# SDK 로드
# =============================================================================

_sdk_lib: Optional[ctypes.CDLL] = None


def _load_sdk() -> ctypes.CDLL:
    """DeckLink SDK 공유 라이브러리를 로드합니다. 캐싱하여 재사용합니다."""
    global _sdk_lib
    if _sdk_lib is not None:
        return _sdk_lib

    system = platform.system()
    if system == 'Darwin':
        lib_path = '/Library/Frameworks/DeckLinkAPI.framework/DeckLinkAPI'
    elif system == 'Linux':
        found = ctypes.util.find_library('DeckLinkAPI')
        lib_path = found if found else '/usr/lib/libDeckLinkAPI.so'
    else:
        raise DeckLinkSDKNotFoundError(
            f"지원하지 않는 플랫폼: {system}. macOS 또는 Linux만 지원합니다."
        )

    try:
        _sdk_lib = ctypes.CDLL(lib_path)
    except OSError as exc:
        raise DeckLinkSDKNotFoundError(
            f"DeckLink SDK를 로드할 수 없습니다: {lib_path}\n"
            "Blackmagic Desktop Video SDK 12.x를 설치하세요.\n"
            "다운로드: https://www.blackmagicdesign.com/support"
        ) from exc

    logger.info(f"DeckLink SDK 로드 완료: {lib_path}")
    return _sdk_lib


def is_sdk_available() -> bool:
    """DeckLink SDK가 설치되어 로드 가능한지 확인합니다."""
    try:
        _load_sdk()
        return True
    except DeckLinkSDKNotFoundError:
        return False


# =============================================================================
# ctypes 기본 타입
# =============================================================================

HRESULT = ctypes.c_int32
ULONG   = ctypes.c_ulong


def _make_vtable_call(
    obj_ptr: int,
    slot: int,
    restype,
    *argtypes,
) -> ctypes.CFUNCTYPE:
    """
    COM 객체의 vtable에서 slot번째 함수 포인터를 가져와 호출 가능한 CFUNCTYPE을 반환합니다.

    COM 객체 메모리 레이아웃:
        [obj_ptr] → vtable_ptr → [fn0, fn1, fn2, ..., fn_slot, ...]
    """
    vtable_ptr = ctypes.cast(obj_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    fn_ptr     = ctypes.cast(vtable_ptr, ctypes.POINTER(ctypes.c_void_p))[slot]
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn_ptr)


# =============================================================================
# IID (Interface Identifier) 정의
# =============================================================================

# IDeckLinkInput: {AF22762B-548A-4B51-AF1C-C5B0AA63F66E}
IID_IDeckLinkInput = (ctypes.c_uint8 * 16)(
    0xAF, 0x22, 0x76, 0x2B,
    0x54, 0x8A,
    0x4B, 0x51,
    0xAF, 0x1C, 0xC5, 0xB0, 0xAA, 0x63, 0xF6, 0x6E,
)


# =============================================================================
# IDeckLinkIterator 래퍼
# =============================================================================

class DeckLinkIterator:
    """
    IDeckLinkIterator COM 인터페이스 래퍼입니다.

    vtable (SDK 12.x):
        [0] QueryInterface  [1] AddRef  [2] Release
        [3] Next
    """

    _SLOT_NEXT = 3

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def Next(self) -> Optional['DeckLinkDevice']:
        """다음 DeckLink 장치를 반환합니다. 없으면 None을 반환합니다."""
        out = ctypes.c_void_p()
        fn  = _make_vtable_call(
            self._ptr, self._SLOT_NEXT,
            HRESULT,
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = fn(self._ptr, ctypes.byref(out))
        if hr != S_OK or not out.value:
            return None
        return DeckLinkDevice(out.value)

    def Release(self) -> None:
        fn = _make_vtable_call(self._ptr, 2, ULONG)
        fn(self._ptr)


# =============================================================================
# IDeckLink 래퍼
# =============================================================================

class DeckLinkDevice:
    """
    IDeckLink COM 인터페이스 래퍼입니다.

    vtable (SDK 12.x):
        [0] QueryInterface  [1] AddRef  [2] Release  ...
    """

    _SLOT_QUERY_INTERFACE = 0

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def GetInput(self) -> 'DeckLinkInput':
        """IDeckLink.QueryInterface(IID_IDeckLinkInput)를 호출하여 IDeckLinkInput을 획득합니다."""
        out = ctypes.c_void_p()
        fn  = _make_vtable_call(
            self._ptr, self._SLOT_QUERY_INTERFACE,
            HRESULT,
            ctypes.c_void_p,                     # REFIID
            ctypes.POINTER(ctypes.c_void_p),     # ppv
        )
        hr = fn(self._ptr, IID_IDeckLinkInput, ctypes.byref(out))
        if hr != S_OK or not out.value:
            raise DeckLinkAPIError(
                f"IDeckLinkInput 인터페이스를 가져올 수 없습니다. "
                f"HRESULT=0x{hr & 0xFFFFFFFF:08X}\n"
                "장치가 입력 모드를 지원하는지 확인하세요."
            )
        return DeckLinkInput(out.value)

    def Release(self) -> None:
        fn = _make_vtable_call(self._ptr, 2, ULONG)
        fn(self._ptr)


# =============================================================================
# IDeckLinkInput 래퍼
# =============================================================================

class DeckLinkInput:
    """
    IDeckLinkInput COM 인터페이스 래퍼입니다. (DeckLink SDK 12.x 기준)

    vtable 슬롯:
        [0]  QueryInterface
        [1]  AddRef
        [2]  Release
        [3]  DoesSupportVideoMode
        [4]  GetDisplayModeIterator
        [5]  EnableVideoInput          ← 사용
        [6]  EnableAudioInput          ← 사용
        [7]  SetCallback               ← 사용
        [8]  GetAvailableVideoFrameCount
        [9]  CreateVideoFrame
        [10] ScheduleVideoFrame
        [11] SetScheduledFrameCompletionCallback
        [12] DisableVideoInput
        [13] DisableAudioInput
        [14] AvailableAudioSampleCount
        [15] CreateAudioSampleBuffer
        [16] ReadAudioSamples
        [17] StartStreams               ← 사용
        [18] StopStreams                ← 사용
        [19] FlushStreams
        [20] PauseStreams
    """

    _SLOT_ENABLE_VIDEO  = 5
    _SLOT_ENABLE_AUDIO  = 6
    _SLOT_SET_CALLBACK  = 7
    _SLOT_START_STREAMS = 17
    _SLOT_STOP_STREAMS  = 18

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def EnableVideoInput(
        self,
        display_mode: int,
        pixel_format: int,
        flags: int = BMDVideoInputFlags.Default,
    ) -> None:
        """비디오 입력을 활성화합니다."""
        fn = _make_vtable_call(
            self._ptr, self._SLOT_ENABLE_VIDEO,
            HRESULT,
            ctypes.c_uint32,  # BMDDisplayMode
            ctypes.c_uint32,  # BMDPixelFormat
            ctypes.c_uint32,  # BMDVideoInputFlags
        )
        hr = fn(self._ptr, display_mode, pixel_format, flags)
        if hr != S_OK:
            raise DeckLinkAPIError(
                f"EnableVideoInput 실패. HRESULT=0x{hr & 0xFFFFFFFF:08X}"
            )

    def EnableAudioInput(
        self,
        sample_rate: int,
        sample_type: int,
        channels: int,
    ) -> None:
        """오디오 입력을 활성화합니다."""
        fn = _make_vtable_call(
            self._ptr, self._SLOT_ENABLE_AUDIO,
            HRESULT,
            ctypes.c_uint32,  # BMDAudioSampleRate
            ctypes.c_uint32,  # BMDAudioSampleType
            ctypes.c_uint32,  # channelCount
        )
        hr = fn(self._ptr, sample_rate, sample_type, channels)
        if hr != S_OK:
            raise DeckLinkAPIError(
                f"EnableAudioInput 실패. HRESULT=0x{hr & 0xFFFFFFFF:08X}"
            )

    def SetCallback(self, callback_ptr: int) -> None:
        """IDeckLinkInputCallback 포인터를 등록합니다."""
        fn = _make_vtable_call(
            self._ptr, self._SLOT_SET_CALLBACK,
            HRESULT,
            ctypes.c_void_p,  # IDeckLinkInputCallback*
        )
        hr = fn(self._ptr, callback_ptr)
        if hr != S_OK:
            raise DeckLinkAPIError(
                f"SetCallback 실패. HRESULT=0x{hr & 0xFFFFFFFF:08X}"
            )

    def StartStreams(self) -> None:
        """비디오/오디오 스트림 캡처를 시작합니다."""
        fn = _make_vtable_call(self._ptr, self._SLOT_START_STREAMS, HRESULT)
        hr = fn(self._ptr)
        if hr != S_OK:
            raise DeckLinkAPIError(
                f"StartStreams 실패. HRESULT=0x{hr & 0xFFFFFFFF:08X}"
            )

    def StopStreams(self) -> None:
        """비디오/오디오 스트림 캡처를 중지합니다. 오류는 무시합니다."""
        fn = _make_vtable_call(self._ptr, self._SLOT_STOP_STREAMS, HRESULT)
        fn(self._ptr)

    def Release(self) -> None:
        fn = _make_vtable_call(self._ptr, 2, ULONG)
        fn(self._ptr)


# =============================================================================
# IDeckLinkVideoInputFrame 래퍼
# =============================================================================

class DeckLinkVideoFrame:
    """
    IDeckLinkVideoInputFrame COM 인터페이스 래퍼입니다.

    vtable 슬롯:
        [0] QueryInterface  [1] AddRef  [2] Release
        [3] GetWidth   [4] GetHeight  [5] GetRowBytes
        [6] GetPixelFormat  [7] GetFlags
        [8] GetBytes   [9] GetTimecode  [10] GetAncillaryData
    """

    _SLOT_GET_WIDTH     = 3
    _SLOT_GET_HEIGHT    = 4
    _SLOT_GET_ROW_BYTES = 5
    _SLOT_GET_BYTES     = 8

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def GetWidth(self) -> int:
        fn = _make_vtable_call(self._ptr, self._SLOT_GET_WIDTH, ctypes.c_long)
        return fn(self._ptr)

    def GetHeight(self) -> int:
        fn = _make_vtable_call(self._ptr, self._SLOT_GET_HEIGHT, ctypes.c_long)
        return fn(self._ptr)

    def GetRowBytes(self) -> int:
        fn = _make_vtable_call(self._ptr, self._SLOT_GET_ROW_BYTES, ctypes.c_long)
        return fn(self._ptr)

    def GetBytes(self) -> bytes:
        """픽셀 데이터를 복사하여 반환합니다. YUV422: 2바이트/픽셀."""
        buf_ptr = ctypes.c_void_p()
        fn = _make_vtable_call(
            self._ptr, self._SLOT_GET_BYTES,
            HRESULT,
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = fn(self._ptr, ctypes.byref(buf_ptr))
        if hr != S_OK or not buf_ptr.value:
            return b''
        row_bytes = self.GetRowBytes()
        height    = self.GetHeight()
        size      = row_bytes * height
        return bytes((ctypes.c_uint8 * size).from_address(buf_ptr.value))


# =============================================================================
# IDeckLinkAudioInputPacket 래퍼
# =============================================================================

class DeckLinkAudioPacket:
    """
    IDeckLinkAudioInputPacket COM 인터페이스 래퍼입니다.

    vtable 슬롯:
        [0] QueryInterface  [1] AddRef  [2] Release
        [3] GetSampleCount  [4] GetBytes  [5] GetPacketTime
    """

    _SLOT_GET_SAMPLE_COUNT = 3
    _SLOT_GET_BYTES        = 4

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def GetSampleCount(self) -> int:
        fn = _make_vtable_call(self._ptr, self._SLOT_GET_SAMPLE_COUNT, ctypes.c_uint32)
        return fn(self._ptr)

    def GetBytes(self, bit_depth: int = 32, channels: int = 2) -> bytes:
        """PCM 데이터를 복사하여 반환합니다."""
        buf_ptr = ctypes.c_void_p()
        fn = _make_vtable_call(
            self._ptr, self._SLOT_GET_BYTES,
            HRESULT,
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = fn(self._ptr, ctypes.byref(buf_ptr))
        if hr != S_OK or not buf_ptr.value:
            return b''
        count             = self.GetSampleCount()
        bytes_per_sample  = bit_depth // 8
        size              = count * channels * bytes_per_sample
        return bytes((ctypes.c_uint8 * size).from_address(buf_ptr.value))


# =============================================================================
# IDeckLinkInputCallback Python 구현체 (역방향 vtable 바인딩)
# =============================================================================

# 콜백 함수 시그니처
_VideoInputFrameArrivedFn = ctypes.CFUNCTYPE(
    HRESULT,
    ctypes.c_void_p,  # this (IDeckLinkInputCallback*)
    ctypes.c_void_p,  # IDeckLinkVideoInputFrame*
    ctypes.c_void_p,  # IDeckLinkAudioInputPacket*
)

_VideoInputFormatChangedFn = ctypes.CFUNCTYPE(
    HRESULT,
    ctypes.c_void_p,  # this
    ctypes.c_uint32,  # BMDVideoInputFormatChangedEvents
    ctypes.c_void_p,  # IDeckLinkDisplayMode*
    ctypes.c_uint32,  # BMDDetectedVideoInputFormatFlags
)

_QueryInterfaceFn = ctypes.CFUNCTYPE(
    HRESULT,
    ctypes.c_void_p,                     # this
    ctypes.c_void_p,                     # REFIID
    ctypes.POINTER(ctypes.c_void_p),     # ppv
)

_AddRefReleaseFn = ctypes.CFUNCTYPE(ULONG, ctypes.c_void_p)


class _DeckLinkCallbackVTable(ctypes.Structure):
    """
    IDeckLinkInputCallback vtable 구조체입니다.

    슬롯 순서 (SDK 12.x):
        [0] QueryInterface  [1] AddRef  [2] Release
        [3] VideoInputFormatChanged
        [4] VideoInputFrameArrived
    """
    _fields_ = [
        ('QueryInterface',          _QueryInterfaceFn),
        ('AddRef',                  _AddRefReleaseFn),
        ('Release',                 _AddRefReleaseFn),
        ('VideoInputFormatChanged', _VideoInputFormatChangedFn),
        ('VideoInputFrameArrived',  _VideoInputFrameArrivedFn),
    ]


class _DeckLinkCallbackObject(ctypes.Structure):
    """COM 객체 메모리 레이아웃 (첫 번째 필드가 vtable 포인터)"""
    _fields_ = [('lpVtbl', ctypes.POINTER(_DeckLinkCallbackVTable))]


class DeckLinkInputCallback:
    """
    IDeckLinkInputCallback의 Python 구현체입니다.

    C++ COM vtable을 직접 구성하여 DeckLink SDK가 Python 함수를 콜백할 수 있게 합니다.

    on_frame_arrived 시그니처:
        (video: DeckLinkVideoFrame | None, audio: DeckLinkAudioPacket | None) -> None

    사용 예시:
        >>> def handler(video, audio): ...
        >>> cb = DeckLinkInputCallback(handler, audio_bit_depth=32, audio_channels=2)
        >>> decklink_input.SetCallback(cb.get_ptr())
    """

    def __init__(
        self,
        on_frame_arrived: Callable,
        audio_bit_depth: int = 32,
        audio_channels: int = 2,
    ) -> None:
        self._on_frame_arrived = on_frame_arrived
        self._audio_bit_depth  = audio_bit_depth
        self._audio_channels   = audio_channels
        self._ref_count        = 1

        # 함수 포인터를 인스턴스 변수에 보관 (Python GC 방지)
        self._qi_fn  = _QueryInterfaceFn(self._query_interface)
        self._ref_fn = _AddRefReleaseFn(self._add_ref)
        self._rel_fn = _AddRefReleaseFn(self._release)
        self._fmt_fn = _VideoInputFormatChangedFn(self._format_changed)
        self._arr_fn = _VideoInputFrameArrivedFn(self._frame_arrived)

        # vtable 구성
        self._vtable = _DeckLinkCallbackVTable(
            QueryInterface=self._qi_fn,
            AddRef=self._ref_fn,
            Release=self._rel_fn,
            VideoInputFormatChanged=self._fmt_fn,
            VideoInputFrameArrived=self._arr_fn,
        )

        # COM 객체 구성
        self._obj = _DeckLinkCallbackObject(ctypes.pointer(self._vtable))

    def get_ptr(self) -> int:
        """COM 객체의 포인터(int)를 반환합니다. DeckLinkInput.SetCallback()에 전달합니다."""
        return ctypes.addressof(self._obj)

    # ── IUnknown 구현 ─────────────────────────────────────────────────────────

    def _query_interface(self, this: int, iid: int, ppv) -> int:
        return 0x80004002  # E_NOINTERFACE

    def _add_ref(self, this: int) -> int:
        self._ref_count += 1
        return self._ref_count

    def _release(self, this: int) -> int:
        self._ref_count = max(0, self._ref_count - 1)
        return self._ref_count

    # ── IDeckLinkInputCallback 구현 ───────────────────────────────────────────

    def _format_changed(
        self, this: int, events: int, display_mode: int, flags: int
    ) -> int:
        logger.info(f"DeckLink 비디오 포맷 변경 감지 (events=0x{events:08X})")
        return S_OK

    def _frame_arrived(
        self, this: int, video_ptr: int, audio_ptr: int
    ) -> int:
        try:
            video = DeckLinkVideoFrame(video_ptr) if video_ptr else None
            audio = DeckLinkAudioPacket(audio_ptr) if audio_ptr else None
            self._on_frame_arrived(video, audio)
        except Exception as exc:
            logger.error(f"DeckLink 콜백 처리 오류: {exc}", exc_info=True)
        return S_OK


# =============================================================================
# 공개 팩토리 함수
# =============================================================================

def create_iterator() -> DeckLinkIterator:
    """
    DeckLinkIterator를 생성합니다.

    반환값:
        DeckLinkIterator: 장치 목록 순회용 이터레이터

    예외:
        DeckLinkSDKNotFoundError: SDK가 설치되지 않았을 때
        DeckLinkAPIError: 이터레이터 생성 실패 시
    """
    lib = _load_sdk()

    try:
        create_fn = lib.CreateDeckLinkIteratorInstance
    except AttributeError as exc:
        raise DeckLinkAPIError(
            "CreateDeckLinkIteratorInstance를 찾을 수 없습니다. "
            "DeckLink SDK 버전을 확인하세요."
        ) from exc

    create_fn.restype  = ctypes.c_void_p
    create_fn.argtypes = []

    ptr = create_fn()
    if not ptr:
        raise DeckLinkAPIError(
            "DeckLinkIterator 생성 실패. DeckLink 장치가 연결되어 있는지 확인하세요."
        )

    return DeckLinkIterator(ptr)


def open_device(device_index: int = 0) -> DeckLinkDevice:
    """
    지정한 인덱스의 DeckLink 장치를 열어 반환합니다.

    파라미터:
        device_index: 장치 인덱스 (0부터 시작)

    반환값:
        DeckLinkDevice: 열린 장치 객체

    예외:
        DeckLinkSDKNotFoundError: SDK가 없을 때
        DeckLinkDeviceNotFoundError: 장치를 찾지 못했을 때
        DeckLinkAPIError: API 호출 실패 시
    """
    iterator = create_iterator()
    try:
        for idx in range(device_index + 1):
            device = iterator.Next()
            if device is None:
                raise DeckLinkDeviceNotFoundError(
                    f"DeckLink 장치 인덱스 {device_index}를 찾을 수 없습니다. "
                    f"연결된 장치 수: {idx}"
                )
            if idx < device_index:
                device.Release()
    finally:
        iterator.Release()

    return device
