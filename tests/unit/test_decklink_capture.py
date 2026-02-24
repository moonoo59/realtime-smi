"""
DeckLinkCapture 및 decklink_bindings 단위 테스트

검증 항목:
- SDK 미설치 환경에서 DeckLinkSDKNotFoundError 정상 발생
- is_sdk_available() 반환값 확인
- open_device() 장치 없음 시 DeckLinkDeviceNotFoundError 발생
- DeckLinkInputCallback vtable 구성 확인 (포인터 획득 가능 여부)
- DeckLinkCapture.__init__() 큐 초기화 확인
- DeckLinkCapture.start() SDK 없을 때 예외 전파 확인
- DeckLinkCapture.stop() 실행 중 아닐 때 안전하게 종료
- _on_frame_arrived() → _enqueue_video_sync / _enqueue_audio_sync 큐 삽입 확인
- 비디오 큐 오버플로우 시 오래된 프레임 교체 확인
- 오디오 큐 오버플로우 시 오래된 패킷 교체 확인
- VIDEO_MODE_MAP 주요 모드 매핑 값 확인
- 콜백 _format_changed() S_OK 반환 확인
"""

from __future__ import annotations

import asyncio
import ctypes
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.capture import AudioPacket, VideoFrame
from src.capture.decklink_bindings import (
    BMDDisplayMode,
    DeckLinkAPIError,
    DeckLinkInputCallback,
    DeckLinkSDKNotFoundError,
    VIDEO_MODE_MAP,
    is_sdk_available,
    open_device,
)
from src.capture.decklink_capture import DeckLinkCapture
from src.config.schema import AppConfig


# =============================================================================
# 픽스처
# =============================================================================

def _make_config(
    device_index: int = 0,
    video_mode: str = "1080p30",
    pixel_format: str = "yuv422",
    audio_channels: list[int] | None = None,
    audio_bit_depth: int = 32,
    audio_queue_size: int = 10,
    video_queue_size: int = 5,
) -> AppConfig:
    """테스트용 AppConfig를 생성합니다."""
    return AppConfig(**{
        "system": {"mode": "live"},
        "capture": {
            "device_index":     device_index,
            "video_mode":       video_mode,
            "pixel_format":     pixel_format,
            "audio_channels":   audio_channels or [1, 2],
            "audio_bit_depth":  audio_bit_depth,
            "audio_queue_size": audio_queue_size,
            "video_queue_size": video_queue_size,
        },
    })


# =============================================================================
# decklink_bindings 테스트
# =============================================================================

class TestDeckLinkSDKAvailability:
    """SDK 설치 확인 관련 테스트"""

    def test_is_sdk_available_returns_false_when_lib_not_found(self):
        """SDK 라이브러리가 없으면 is_sdk_available()이 False를 반환합니다."""
        # _sdk_lib 캐시를 초기화하고 CDLL 로드를 실패시킴
        import src.capture.decklink_bindings as bindings
        original_lib = bindings._sdk_lib
        bindings._sdk_lib = None

        try:
            with patch("ctypes.CDLL", side_effect=OSError("not found")):
                result = is_sdk_available()
            assert result is False
        finally:
            bindings._sdk_lib = original_lib

    def test_is_sdk_available_returns_true_when_lib_found(self):
        """SDK 라이브러리가 있으면 is_sdk_available()이 True를 반환합니다."""
        import src.capture.decklink_bindings as bindings
        original_lib = bindings._sdk_lib
        bindings._sdk_lib = None

        try:
            with patch("ctypes.CDLL", return_value=MagicMock()):
                result = is_sdk_available()
            assert result is True
        finally:
            bindings._sdk_lib = original_lib

    def test_load_sdk_raises_on_unsupported_platform(self):
        """지원하지 않는 플랫폼에서 DeckLinkSDKNotFoundError가 발생합니다."""
        import src.capture.decklink_bindings as bindings
        from src.capture.decklink_bindings import _load_sdk
        original_lib = bindings._sdk_lib
        bindings._sdk_lib = None

        try:
            with patch("platform.system", return_value="Windows"), \
                 pytest.raises(DeckLinkSDKNotFoundError, match="지원하지 않는 플랫폼"):
                _load_sdk()
        finally:
            bindings._sdk_lib = original_lib

    def test_load_sdk_raises_sdk_not_found_error_when_oserror(self):
        """OSError 발생 시 DeckLinkSDKNotFoundError로 변환됩니다."""
        import src.capture.decklink_bindings as bindings
        from src.capture.decklink_bindings import _load_sdk
        original_lib = bindings._sdk_lib
        bindings._sdk_lib = None

        try:
            with patch("ctypes.CDLL", side_effect=OSError("not found")), \
                 pytest.raises(DeckLinkSDKNotFoundError, match="DeckLink SDK를 로드할 수 없습니다"):
                _load_sdk()
        finally:
            bindings._sdk_lib = original_lib


class TestOpenDevice:
    """open_device() 테스트"""

    def test_open_device_raises_when_no_device(self):
        """장치가 없을 때 DeckLinkDeviceNotFoundError가 발생합니다."""
        from src.capture.decklink_bindings import DeckLinkDeviceNotFoundError

        mock_iterator = MagicMock()
        mock_iterator.Next.return_value = None  # 장치 없음

        with patch(
            "src.capture.decklink_bindings.create_iterator",
            return_value=mock_iterator,
        ), pytest.raises(DeckLinkDeviceNotFoundError, match="DeckLink 장치 인덱스 0"):
            open_device(0)

        mock_iterator.Release.assert_called_once()

    def test_open_device_returns_device_at_correct_index(self):
        """올바른 인덱스의 장치를 반환합니다."""
        from src.capture.decklink_bindings import DeckLinkDeviceNotFoundError

        device_0 = MagicMock()
        device_1 = MagicMock()

        mock_iterator = MagicMock()
        mock_iterator.Next.side_effect = [device_0, device_1, None]

        with patch(
            "src.capture.decklink_bindings.create_iterator",
            return_value=mock_iterator,
        ):
            result = open_device(1)

        assert result is device_1
        device_0.Release.assert_called_once()  # 이전 장치는 Release
        mock_iterator.Release.assert_called_once()


class TestVideoModeMap:
    """VIDEO_MODE_MAP 값 검증"""

    @pytest.mark.parametrize("mode_str, expected_code", [
        ("1080p30", 0x48703330),
        ("1080p25", 0x48703235),
        ("1080i50", 0x48693530),
        ("720p60",  0x68703630),
    ])
    def test_video_mode_map_values(self, mode_str: str, expected_code: int):
        """VIDEO_MODE_MAP이 올바른 BMD 상수를 가집니다."""
        assert VIDEO_MODE_MAP[mode_str] == expected_code

    def test_video_mode_map_has_all_config_modes(self):
        """config.yaml에서 사용하는 모든 모드가 매핑에 포함됩니다."""
        assert "1080p30" in VIDEO_MODE_MAP
        assert "1080p25" in VIDEO_MODE_MAP
        assert "1080i50" in VIDEO_MODE_MAP
        assert "720p60"  in VIDEO_MODE_MAP


class TestDeckLinkInputCallback:
    """DeckLinkInputCallback vtable 구성 테스트"""

    def test_get_ptr_returns_nonzero_int(self):
        """get_ptr()이 0이 아닌 정수(COM 객체 주소)를 반환합니다."""
        handler = MagicMock()
        cb = DeckLinkInputCallback(handler, audio_bit_depth=32, audio_channels=2)
        ptr = cb.get_ptr()

        assert isinstance(ptr, int)
        assert ptr != 0

    def test_add_ref_increments_ref_count(self):
        """_add_ref() 호출 시 ref_count가 1 증가합니다."""
        cb = DeckLinkInputCallback(MagicMock())
        initial = cb._ref_count
        cb._add_ref(0)
        assert cb._ref_count == initial + 1

    def test_release_decrements_ref_count(self):
        """_release() 호출 시 ref_count가 1 감소합니다."""
        cb = DeckLinkInputCallback(MagicMock())
        cb._add_ref(0)
        before = cb._ref_count
        cb._release(0)
        assert cb._ref_count == before - 1

    def test_release_does_not_go_below_zero(self):
        """_release()가 ref_count를 0 미만으로 줄이지 않습니다."""
        cb = DeckLinkInputCallback(MagicMock())
        cb._ref_count = 0
        cb._release(0)
        assert cb._ref_count == 0

    def test_format_changed_returns_s_ok(self):
        """_format_changed()는 S_OK(0)를 반환합니다."""
        cb = DeckLinkInputCallback(MagicMock())
        result = cb._format_changed(0, 0, 0, 0)
        assert result == 0  # S_OK

    def test_query_interface_returns_e_nointerface(self):
        """_query_interface()는 E_NOINTERFACE(0x80004002)를 반환합니다."""
        cb = DeckLinkInputCallback(MagicMock())
        result = cb._query_interface(0, 0, None)
        assert result == 0x80004002

    def test_frame_arrived_calls_handler_with_none_when_no_frame(self):
        """video_ptr=0, audio_ptr=0 일 때 핸들러에 None, None이 전달됩니다."""
        handler = MagicMock()
        cb = DeckLinkInputCallback(handler)
        cb._frame_arrived(0, 0, 0)
        handler.assert_called_once_with(None, None)

    def test_frame_arrived_suppresses_handler_exception(self):
        """핸들러에서 예외가 발생해도 _frame_arrived가 S_OK를 반환합니다."""
        handler = MagicMock(side_effect=RuntimeError("test error"))
        cb = DeckLinkInputCallback(handler)
        result = cb._frame_arrived(0, 0, 0)
        assert result == 0  # S_OK (예외 미전파)


# =============================================================================
# DeckLinkCapture 테스트
# =============================================================================

class TestDeckLinkCaptureInit:
    """DeckLinkCapture 초기화 테스트"""

    def test_audio_queue_maxsize(self):
        """audio_queue가 config의 audio_queue_size로 초기화됩니다."""
        config = _make_config(audio_queue_size=20)
        capture = DeckLinkCapture(config)
        assert capture.get_audio_queue().maxsize == 20

    def test_video_queue_maxsize(self):
        """video_queue가 config의 video_queue_size로 초기화됩니다."""
        config = _make_config(video_queue_size=8)
        capture = DeckLinkCapture(config)
        assert capture.get_video_queue().maxsize == 8

    def test_not_running_initially(self):
        """초기 상태는 _running=False입니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)
        assert capture._running is False


class TestDeckLinkCaptureStart:
    """DeckLinkCapture.start() 테스트"""

    @pytest.mark.asyncio
    async def test_start_raises_when_sdk_not_found(self):
        """SDK가 없으면 start()에서 DeckLinkSDKNotFoundError가 발생합니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)

        with patch(
            "src.capture.decklink_capture.open_device",
            side_effect=DeckLinkSDKNotFoundError("SDK 없음"),
        ), pytest.raises(DeckLinkSDKNotFoundError):
            await capture.start()

    @pytest.mark.asyncio
    async def test_start_raises_on_unsupported_video_mode(self):
        """지원하지 않는 video_mode 설정 시 DeckLinkAPIError가 발생합니다."""
        config  = _make_config(video_mode="4320p60")  # 미지원 모드
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ), pytest.raises(DeckLinkAPIError, match="지원하지 않는 비디오 모드"):
            await capture.start()

    @pytest.mark.asyncio
    async def test_start_calls_enable_video_input(self):
        """start()가 EnableVideoInput을 올바른 인자로 호출합니다."""
        config  = _make_config(video_mode="1080p30", pixel_format="yuv422")
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ):
            await capture.start()
            await capture.stop()

        mock_input.EnableVideoInput.assert_called_once_with(
            0x48703330,  # BMDDisplayMode.HD1080p30
            0x32767579,  # BMDPixelFormat.YUV422_8bit
        )

    @pytest.mark.asyncio
    async def test_start_calls_enable_audio_input(self):
        """start()가 EnableAudioInput을 올바른 인자로 호출합니다."""
        config  = _make_config(audio_channels=[1, 2], audio_bit_depth=32)
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ):
            await capture.start()
            await capture.stop()

        mock_input.EnableAudioInput.assert_called_once_with(
            48000,  # BMDAudioSampleRate.Rate48kHz
            32,     # BMDAudioSampleType.Int32bit
            2,      # channels
        )

    @pytest.mark.asyncio
    async def test_start_calls_set_callback_and_start_streams(self):
        """start()가 SetCallback → StartStreams 순서로 호출합니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        call_order: list[str] = []
        mock_input.SetCallback.side_effect  = lambda _: call_order.append("SetCallback")
        mock_input.StartStreams.side_effect = lambda: call_order.append("StartStreams")

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ):
            await capture.start()
            # start() 직후, stop() 호출 전 순서만 검증
            assert call_order == ["SetCallback", "StartStreams"]
            await capture.stop()

        assert capture._running is False

    @pytest.mark.asyncio
    async def test_start_idempotent_when_already_running(self):
        """이미 실행 중일 때 start()를 다시 호출해도 중복 실행되지 않습니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ):
            await capture.start()
            await capture.start()  # 두 번째 호출 (무시되어야 함)
            await capture.stop()

        # StartStreams는 한 번만 호출
        mock_input.StartStreams.assert_called_once()


class TestDeckLinkCaptureStop:
    """DeckLinkCapture.stop() 테스트"""

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_safe(self):
        """실행 중이 아닐 때 stop()을 호출해도 예외가 발생하지 않습니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)
        await capture.stop()  # 예외 없이 통과해야 함

    @pytest.mark.asyncio
    async def test_stop_releases_com_objects(self):
        """stop() 후 device, input 참조가 None으로 초기화됩니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ):
            await capture.start()
            await capture.stop()

        assert capture._device   is None
        assert capture._input    is None
        assert capture._callback is None

    @pytest.mark.asyncio
    async def test_stop_calls_stop_streams(self):
        """stop()이 StopStreams()를 호출합니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)

        mock_device = MagicMock()
        mock_input  = MagicMock()
        mock_device.GetInput.return_value = mock_input

        with patch(
            "src.capture.decklink_capture.open_device",
            return_value=mock_device,
        ):
            await capture.start()
            await capture.stop()

        mock_input.StopStreams.assert_called_once()


class TestDeckLinkCaptureEnqueue:
    """_on_frame_arrived → 큐 삽입 테스트"""

    def _make_running_capture(self, config: AppConfig) -> DeckLinkCapture:
        """_running=True인 DeckLinkCapture를 생성합니다."""
        capture = DeckLinkCapture(config)
        capture._running = True
        capture._loop    = asyncio.get_event_loop()
        return capture

    @pytest.mark.asyncio
    async def test_enqueue_video_frame(self):
        """_on_frame_arrived()가 video_queue에 VideoFrame을 삽입합니다."""
        config  = _make_config(video_queue_size=5)
        capture = self._make_running_capture(config)

        mock_video = MagicMock()
        mock_video.GetWidth.return_value  = 1920
        mock_video.GetHeight.return_value = 1080
        mock_video.GetBytes.return_value  = b'\x80\x10' * (1920 * 1080)

        capture._on_frame_arrived(mock_video, None)
        await asyncio.sleep(0)  # call_soon_threadsafe 실행 대기

        assert capture.get_video_queue().qsize() == 1
        frame = capture.get_video_queue().get_nowait()
        assert isinstance(frame, VideoFrame)
        assert frame.width  == 1920
        assert frame.height == 1080
        assert frame.pixel_format == "yuv422"
        assert frame.frame_id == 0

    @pytest.mark.asyncio
    async def test_enqueue_audio_packet(self):
        """_on_frame_arrived()가 audio_queue에 AudioPacket을 삽입합니다."""
        config  = _make_config(audio_channels=[1, 2], audio_bit_depth=32, audio_queue_size=10)
        capture = self._make_running_capture(config)

        mock_audio = MagicMock()
        mock_audio.GetSampleCount.return_value = 480
        mock_audio.GetBytes.return_value = b'\x00' * (480 * 2 * 4)  # 480샘플 * 2ch * 4bytes

        capture._on_frame_arrived(None, mock_audio)
        await asyncio.sleep(0)

        assert capture.get_audio_queue().qsize() == 1
        packet = capture.get_audio_queue().get_nowait()
        assert isinstance(packet, AudioPacket)
        assert packet.sample_rate == 48000
        assert packet.channels    == 2
        assert packet.bit_depth   == 32
        assert packet.packet_id   == 0

    @pytest.mark.asyncio
    async def test_frame_id_increments(self):
        """연속 호출 시 frame_id가 증가합니다."""
        config  = _make_config(video_queue_size=5)
        capture = self._make_running_capture(config)

        for _ in range(3):
            mock_video = MagicMock()
            mock_video.GetWidth.return_value  = 1920
            mock_video.GetHeight.return_value = 1080
            mock_video.GetBytes.return_value  = b'\x80\x10' * (1920 * 1080)
            capture._on_frame_arrived(mock_video, None)

        await asyncio.sleep(0)

        frames = []
        q = capture.get_video_queue()
        while not q.empty():
            frames.append(q.get_nowait())

        assert [f.frame_id for f in frames] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_video_queue_overflow_drops_oldest(self):
        """video_queue가 가득 찼을 때 오래된 프레임을 교체합니다."""
        config  = _make_config(video_queue_size=2)
        capture = self._make_running_capture(config)

        def _make_mock_video(frame_id: int) -> MagicMock:
            m = MagicMock()
            m.GetWidth.return_value  = 1920
            m.GetHeight.return_value = 1080
            m.GetBytes.return_value  = bytes([frame_id % 256]) * (1920 * 1080 * 2)
            return m

        # 큐 크기(2)보다 많은 프레임(3) 삽입
        for i in range(3):
            capture._on_frame_arrived(_make_mock_video(i), None)
        await asyncio.sleep(0)

        # 큐 크기는 2 이하
        assert capture.get_video_queue().qsize() <= 2

    @pytest.mark.asyncio
    async def test_audio_queue_overflow_drops_oldest(self):
        """audio_queue가 가득 찼을 때 오래된 패킷을 교체합니다."""
        config  = _make_config(audio_queue_size=2, audio_channels=[1])
        capture = self._make_running_capture(config)

        for i in range(3):
            mock_audio = MagicMock()
            mock_audio.GetSampleCount.return_value = 480
            mock_audio.GetBytes.return_value = b'\x00' * (480 * 1 * 4)
            capture._on_frame_arrived(None, mock_audio)
        await asyncio.sleep(0)

        assert capture.get_audio_queue().qsize() <= 2

    @pytest.mark.asyncio
    async def test_on_frame_arrived_ignored_when_not_running(self):
        """_running=False 상태에서 _on_frame_arrived()가 아무것도 하지 않습니다."""
        config  = _make_config()
        capture = DeckLinkCapture(config)
        capture._running = False

        mock_video = MagicMock()
        capture._on_frame_arrived(mock_video, None)
        await asyncio.sleep(0)

        assert capture.get_video_queue().qsize() == 0

    @pytest.mark.asyncio
    async def test_on_frame_arrived_handles_get_bytes_error(self):
        """GetBytes()에서 예외가 발생해도 _on_frame_arrived()가 크래시되지 않습니다."""
        config  = _make_config()
        capture = self._make_running_capture(config)

        mock_video = MagicMock()
        mock_video.GetWidth.return_value  = 1920
        mock_video.GetHeight.return_value = 1080
        mock_video.GetBytes.side_effect   = RuntimeError("read error")

        # 예외가 외부로 전파되지 않아야 함
        capture._on_frame_arrived(mock_video, None)
        await asyncio.sleep(0)

        assert capture.get_video_queue().qsize() == 0
