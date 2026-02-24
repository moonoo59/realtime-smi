# SDI-RealtimeSubtitle — 프로젝트 진행 현황

```
문서명:    프로젝트 진행 현황 (Project Status Report)
버전:      1.1.0
작성일:    2026-02-23
최종 수정: 2026-02-24
작성자:    Claude Code (Sonnet 4.6)
상태:      승인됨
```

## 변경 이력

| 버전  | 날짜       | 변경 내용              | 작성자       |
|-------|------------|------------------------|--------------|
| 1.0.0 | 2026-02-23 | 최초 작성 (Phase 1~2 완료 시점 스냅샷) | Claude Code |
| 1.1.0 | 2026-02-24 | Phase 3 Step 1~2, 4 완료 반영 (BUG-001~004 수정, 웹 대시보드, 핫스왑 설정) | Claude Code |
| 1.2.0 | 2026-02-24 | Phase 3 Step 3 완료 (DeckLink SDK ctypes 바인딩, 실 캡처 구현, 38개 테스트 추가) | Claude Code |

---

## 1. 전체 진행률 요약

| Phase | 명칭                        | 상태      | 완료율 |
|-------|-----------------------------|-----------|--------|
| 1     | 핵심 파이프라인             | 완료      | 100%   |
| 2     | 품질/모니터링 레이어        | 완료      | 100%   |
| 3     | DeckLink 실 연동 / 웹 대시보드 | 완료      | 100%   |
| 4     | 다중 채널 / 운영 최적화     | 미착수    | 0%     |

**전체 진행률: 약 80%**

> Phase 1~2는 테스트 통과로 검증 완료.
> Phase 3 전체 완료 (BUG 수정·웹 대시보드·핫스왑·DeckLink SDK 실 연동).
> 현재 테스트: **291 passed, 2 skipped**.

---

## 2. 완료된 모듈 목록

### 2-1. Phase 1 — 핵심 파이프라인

| 모듈 경로 | 역할 | 테스트 |
|-----------|------|--------|
| `src/config/schema.py` | Pydantic v2 기반 AppConfig 스키마 정의 | 통과 |
| `src/config/config_manager.py` | YAML 로드, 환경변수 오버라이드, watchdog 핫스왑 감지 | 통과 |
| `src/capture/file_mock_capture.py` | WAV 파일 기반 오디오/비디오 모의 캡처 (loop, playback_speed 파라미터 지원) | 통과 |
| `src/capture/decklink_capture.py` | DeckLink SDK stub (Phase 3 Step 3에서 실 구현 예정) | — |
| `src/audio/resampler.py` | 48kHz/24bit → 16kHz/16bit/mono 리샘플링, VAD 필터, RMS/Peak 계산, **핫스왑 update_config()** | 통과 |
| `src/stt/clova_streamer.py` | Clova Speech gRPC 양방향 스트리밍, exponential backoff 재연결 (최대 5회) | 통과 |
| `src/subtitle/subtitle_manager.py` | partial/final 이벤트 관리, sync_offset 적용, **핫스왑 update_config()** | 통과 |
| `src/subtitle/subtitle_exporter.py` | SRT/VTT 파일 내보내기 (세션 기준 상대 타임코드) | 통과 |
| `src/compositor/video_compositor.py` | Pillow 한글 렌더링, OpenCV 오버레이, 프레임 드롭 감지, **핫스왑 update_style()** | 통과 |
| `main.py` | asyncio 파이프라인 오케스트레이터, **Pipeline.apply_config() 핫스왑 연결** | — |

### 2-2. Phase 2 — 품질/모니터링 레이어

| 모듈 경로 | 역할 | 테스트 |
|-----------|------|--------|
| `src/metrics/latency_tracker.py` | 단계별 지연시간 기록, 60초 슬라이딩 윈도우 P95/P99, CSV 내보내기 | 통과 |
| `src/metrics/accuracy_evaluator.py` | jiwer 기반 WER/CER 계산, JSON 리포트 생성 | 통과 |
| `src/metrics/metrics_store.py` | 오디오 레벨·STT 상태·프레임 통계 중앙 저장소 (thread-safe RLock) | 통과 |
| `src/logging/structured_logger.py` | JSON 구조화 로깅, 세션 ID 추적, 로테이션 | 통과 |
| `src/dashboard/tui_dashboard.py` | Textual 기반 TUI 6패널 대시보드 (500ms 폴링) | 통과 |

### 2-3. Phase 3 (완료 항목)

| 모듈 경로 | 역할 | 테스트 |
|-----------|------|--------|
| `src/dashboard/web_dashboard.py` | FastAPI + WebSocket 실시간 메트릭 대시보드, 5종 알림 (`STT_DISCONNECTED`, `HIGH_DROP_RATE`, `HIGH_E2E_LATENCY`, `AUDIO_SILENCE`, `STT_ERROR_BURST`) | 통과 |
| `src/capture/decklink_bindings.py` | DeckLink SDK 12.x ctypes COM vtable 바인딩, `IDeckLinkIterator/Device/Input/VideoFrame/AudioPacket` 래퍼, `DeckLinkInputCallback` Python 역방향 vtable 구현 | 통과 |
| `src/capture/decklink_capture.py` | SDK 실 캡처 구현, 콜백→asyncio Queue 스레드 안전 전달 (`loop.call_soon_threadsafe`), graceful shutdown | 통과 |

### 2-4. 파이프라인 데이터 흐름 (현재 구현 범위)

```
[FileMockCapture / DeckLinkCapture(stub)]
   │ audio_queue           │ video_queue
   ▼                        ▼
[AudioResampler]         [VideoCompositor] ◀─────────────────┐
   │ update_audio_level()   │ update_frame_stats()            │
   │ record_capture()       ▼                        [SubtitleEvent]
   │                  [MetricsStore] ←───────────── [SubtitleManager]
   ▼                        ▲                                 ▲
[ClovaSpeechStreamer]        │ update_stt_status()            │
   │ result_queue            │ update_subtitle()              │
   └──────────────► [_stt_result_consumer] ──────────────────┘
                        │ record_stt_receive()
                        ▼
                  [LatencyTracker]

[MetricsStore] ──→ [TuiDashboard] (500ms 폴링)
             └──→ [WebDashboard] (1초 WebSocket broadcast, --web-dashboard 플래그)

[ConfigManager] ──watch()──→ 파일 변경 감지
             └──subscribe()──→ Pipeline.apply_config()
                                 ├── AudioResampler.update_config()
                                 ├── SubtitleManager.update_config()
                                 └── VideoCompositor.update_style()
```

---

## 3. 버그 현황

> 우선순위 구분: [Critical] 기능 불가 / [Major] 기능 저하 / [Minor] 성능·품질 저하

### ~~BUG-001~~ [Critical] — **수정 완료**

MetricsStore가 Pipeline에 연결되지 않던 문제.
`main.py`의 `_audio_pipeline`, `_video_pipeline`, `_stt_result_consumer`에 `MetricsStore.update_*()` 및 `LatencyTracker.record_*()` 호출 추가 완료.

### ~~BUG-002~~ [Major] — **수정 완료**

SRT/VTT 타임코드가 epoch 절대시간을 사용하던 문제.
`subtitle_exporter.py`의 `export_srt()` / `export_vtt()`에서 첫 자막의 `display_at_ns`를 기준 시각으로 자동 계산하여 상대 타임코드로 변환.

### ~~BUG-003~~ [Major] — **수정 완료**

`STTResult.result_id`가 항상 0이던 문제.
`clova_streamer.py`의 `_parse_response()`에 `self._result_id += 1` 추가 완료.

### ~~BUG-004~~ [Major] — **수정 완료**

시그널 핸들러가 asyncio-safe하지 않던 문제.
`main.py`에서 `signal.signal()` → `loop.add_signal_handler()` 교체 완료.

### ~~BUG-005~~ [Minor] — **수정 완료**

subtitle_manager의 `_auto_export()`가 final 이벤트마다 전체 히스토리를 재기록하던 문제.
`SubtitleExporter`에 `append_srt()`/`append_vtt()` 증분 쓰기 메서드 추가, `SubtitleManager`에
`_total_finals`·`_written_counts`·`_session_start_ns` 추적 변수 추가로 새 항목만 append하도록 수정.
히스토리 오버플로우 시에도 정확한 인덱스 계산 보장. 테스트 5개 추가.

### ~~BUG-006~~ [Minor] — **수정 완료**

`video_compositor.py`의 `_draw_text_pil()`이 외곽선을 루프로 반복 렌더링하던 문제.
Pillow 내장 `draw.text(..., stroke_width=N, stroke_fill=color)` 파라미터를 사용한
단일 호출(O(1))로 구현되어 있음을 확인. 동작 보장을 위해 테스트 1개 추가.

---

## 4. Phase 3 구현 현황

| Step | 내용 | 상태 |
|------|------|------|
| Step 1 | 버그 수정 (BUG-001~004) | **완료** |
| Step 2 | 웹 대시보드 (`web_dashboard.py`) | **완료** |
| Step 3 | DeckLink SDK 실 연동 | **완료** |
| Step 4 | 핫스왑 설정 완성 | **완료** |

### Step 3 — DeckLink SDK 실 연동 (완료)

#### 구현 내용

**`src/capture/decklink_bindings.py`**
- SDK 설치 감지: macOS(`DeckLinkAPI.framework`) / Linux(`libDeckLinkAPI.so`)
- `DeckLinkSDKNotFoundError`, `DeckLinkDeviceNotFoundError`, `DeckLinkAPIError` 예외 계층
- `BMDDisplayMode`, `BMDPixelFormat`, `BMDAudioSampleRate/Type` 상수 (SDK 12.x 기준)
- `_make_vtable_call()` — COM 객체 vtable 슬롯 함수 포인터 획득 유틸리티
- `DeckLinkIterator`, `DeckLinkDevice`, `DeckLinkInput` — 각 COM 인터페이스 래퍼
- `DeckLinkVideoFrame`, `DeckLinkAudioPacket` — 프레임/패킷 데이터 접근 래퍼
- `DeckLinkInputCallback` — Python 역방향 vtable 구현 (ctypes 함수 포인터 + GC 방지)
- `is_sdk_available()`, `open_device(index)` — 공개 팩토리

**`src/capture/decklink_capture.py`**
- `DeckLinkCapture.start()`: 장치 열기 → EnableVideoInput/AudioInput → SetCallback → StartStreams
- `_on_frame_arrived()`: SDK 콜백 스레드에서 호출, `loop.call_soon_threadsafe()`로 asyncio 전달
- `_enqueue_video_sync()` / `_enqueue_audio_sync()`: 큐 오버플로우 시 oldest 교체
- `DeckLinkCapture.stop()`: StopStreams → SetCallback(0) → COM Release → 참조 해제
- YUV422 포맷: `GetRowBytes() × GetHeight()` 기반으로 바이트 수 계산 (VideoCompositor 연동 확인)

**`tests/unit/test_decklink_capture.py`**
- 38개 단위 테스트 (SDK mock 기반, SDK 없이 전부 통과)

#### 실장비 검증 절차 (SDK 설치 후)
1. `pip install` 불필요 (ctypes 기반)
2. Blackmagic Desktop Video SDK 12.x 설치
3. SDI 신호 연결 후 `python3 main.py --mode live --web-dashboard` 실행
4. http://localhost:8765 에서 오디오/비디오 메트릭 확인

---

## 5. Phase별 잔여 작업량

### Phase 3 잔여 작업

| 작업 항목 | 상태 | 우선순위 |
|-----------|------|----------|
| ~~BUG-001~004 수정~~ | 완료 | — |
| ~~웹 대시보드 구현~~ | 완료 | — |
| ~~핫스왑 설정 완성~~ | 완료 | — |
| ~~DeckLink SDK 실 연동~~ | **완료** | — |
| ~~BUG-005, 006 수정 (Minor)~~ | **완료** | — |

### Phase 4 잔여 작업

| 작업 항목 | 예상 공수 | 비고 |
|-----------|-----------|------|
| 다중 오디오 채널 지원 (SDI 채널 1~16) | 4~5일 | 설계 변경 수반 |
| 성능 최적화 (C 확장 검토, GIL 우회) | 3~5일 | 프로파일링 선행 |
| NDI/SDI 재출력 모듈 | 5~7일 | 별도 SDK 필요 |
| 운영 문서화 (배포 가이드, 런북) | 2~3일 | — |
| 8시간 무중단 운영 검증 | 2~3일 | 실장비 필요 |
| **Phase 4 합계** | **약 16~23일** | — |

---

## 6. 기술 스택 참조

| 분류 | 기술 | 버전 |
|------|------|------|
| 런타임 | Python | 3.11+ |
| 비동기 | asyncio | 표준 라이브러리 |
| 설정 검증 | Pydantic | v2.5.0+ |
| gRPC | grpcio / grpcio-tools | 1.60.0+ |
| 오디오 처리 | scipy, soundfile, webrtcvad | — |
| 비디오 처리 | OpenCV | 4.8.0+ |
| 한글 렌더링 | Pillow | 10.0.0+ |
| 정확도 측정 | jiwer | 3.0.0+ |
| TUI 대시보드 | Textual + Rich | 0.47.0+ |
| 웹 대시보드 | FastAPI + uvicorn | 0.110.0+ / 0.27.0+ |
| 로깅 | python-json-logger | 2.0.7+ |
| 설정 파일 감시 | watchdog | 3.0.0+ |

---

## 7. 성공 기준 (설계 명세 기준)

| 항목 | 목표 | 현재 측정 가능 여부 |
|------|------|---------------------|
| End-to-end 자막 지연 | 2,000ms 이하 (목표 1,000ms) | **측정 가능** (MetricsStore 연결 완료) |
| 오디오 캡처 → STT 전송 지연 | 200ms 이하 | **측정 가능** (LatencyTracker 연결 완료) |
| 프레임 드롭율 | 0.1% 이하 (1080p 30fps) | **측정 가능** (MetricsStore 연결 완료) |
| STT 연결 성공률 | 99% 이상 | 로그 및 MetricsStore로 확인 가능 |
| WER 측정 오차 | ±0.5% 이내 | AccuracyEvaluator로 확인 가능 |
| 연속 운영 안정성 | 8시간 무중단 | Phase 4에서 검증 |

---

## 8. 작업 시작 체크리스트

작업을 이어받는 개발자는 다음 순서로 환경을 확인하고 작업을 시작합니다.

```bash
# 1. 의존성 설치 확인
cd /Users/admin/subtitle
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. 기존 테스트 전체 실행 (291 passed, 2 skipped 기준)
python3 -m pytest tests/ -v

# 3. 파일 모드로 파이프라인 동작 확인 (STT 없이)
python3 main.py --mode file --no-stt --no-display --duration 10

# 4. 웹 대시보드 동작 확인 (브라우저에서 http://localhost:8765 접속)
python3 main.py --mode file --no-stt --no-display --web-dashboard --duration 30

# 5. 핫스왑 동작 확인
#    파이프라인 실행 중 config.yaml의 subtitle.font.size 등을 변경하면
#    재시작 없이 즉시 반영됨

# 6. 다음 작업: DeckLink SDK 실 연동
#    파일: src/capture/decklink_capture.py
#    Blackmagic Desktop Video SDK 12.x 설치 후 진행
```

---

*본 문서는 Phase 3 전체 완료 시점(2026-02-24)의 스냅샷입니다. Phase 4 착수 시 버전을 1.3.0으로 갱신하십시오.*
