# SDI-RealtimeSubtitle

SDI 영상 신호에서 오디오를 캡처하여 실시간으로 자막을 생성하고 영상에 오버레이하는 파이프라인입니다.
Clova Speech gRPC 스트리밍 STT를 사용하며, DeckLink SDK를 통해 전문 방송 장비와 연동됩니다.

---

## 주요 기능

- **실시간 STT**: Clova Speech gRPC 양방향 스트리밍, partial/final 결과 처리
- **자막 오버레이**: Pillow 한글 렌더링 + OpenCV 프레임 합성
- **DeckLink 캡처**: Blackmagic DeckLink SDK ctypes 바인딩, YUV422/BGRA 지원
- **파일 테스트 모드**: 실 장비 없이 WAV 파일로 파이프라인 검증 가능
- **자막 내보내기**: SRT/VTT 증분 쓰기 (O(1) I/O)
- **핫스왑 설정**: 서비스 중단 없이 폰트·위치·싱크 오프셋 실시간 변경
- **모니터링 대시보드**: TUI (Textual) 및 Web (FastAPI + WebSocket) 지원
- **메트릭 수집**: 지연시간 P95/P99, 프레임 드롭율, WER/CER 측정

---

## 시스템 요구사항

| 항목 | 요구사항 |
|------|----------|
| OS | macOS / Linux |
| Python | 3.11+ |
| DeckLink SDK | Blackmagic Desktop Video SDK 12.x (live 모드 시) |
| 메모리 | 4GB 이상 권장 |

---

## 설치

```bash
# 저장소 클론
git clone https://github.com/moonoo59/realtime-smi.git
cd realtime-smi
```

### 1. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 테스트 실행 시
```

### 3. 환경변수 설정 (API 키)

```bash
cp .env.example .env
# .env 파일 열고 SRS_STT_SECRET_KEY 입력
source .env
```

---

## 설정

`config.yaml`에서 모든 설정을 관리합니다. 민감한 값은 환경변수로 오버라이드할 수 있습니다.

```bash
export SRS_STT_API_KEY=your_clova_api_key
export SRS_STT_SECRET_KEY=your_clova_secret_key
```

주요 설정 항목:

```yaml
system:
  mode: "live"              # "live" | "file"

stt:
  endpoint: "clovaspeech-gw.ncloud.com:50051"
  language: "ko-KR"

subtitle:
  sync_offset_ms: 0         # 자막 싱크 오프셋 (ms)
  display_duration_ms: 3000 # 자막 표시 유지 시간
  font:
    path: "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    size: 36
```

---

## 실행

### 4. 파일 모드 실행 (장비 없이 테스트)

```bash
python3 main.py --mode file --no-stt --no-display --web-dashboard
# 브라우저에서 http://localhost:8765 접속
# 웹 대시보드의 "파이프라인 제어" 패널에서 파일 경로 입력 후 시작
```

### 5. 라이브 모드 실행 (DeckLink 장비 연결)

```bash
source .env
python3 main.py --mode live --web-dashboard
```

---

## 테스트

```bash
python3 -m pytest tests/ -v
# 291 passed, 2 skipped
```

---

## 프로젝트 구조

```
.
├── main.py                        # asyncio 파이프라인 오케스트레이터
├── config.yaml                    # 전체 설정 파일
├── src/
│   ├── capture/
│   │   ├── decklink_bindings.py   # DeckLink SDK ctypes 바인딩
│   │   ├── decklink_capture.py    # DeckLink 실 캡처 구현
│   │   └── file_mock_capture.py   # 파일 기반 테스트 캡처
│   ├── audio/
│   │   └── resampler.py           # 48kHz/24bit → 16kHz/16bit 리샘플링, VAD
│   ├── stt/
│   │   └── clova_streamer.py      # Clova Speech gRPC 스트리밍 클라이언트
│   ├── subtitle/
│   │   ├── subtitle_manager.py    # partial/final 자막 이벤트 관리
│   │   └── subtitle_exporter.py   # SRT/VTT 파일 내보내기
│   ├── compositor/
│   │   └── video_compositor.py    # Pillow 렌더링 + OpenCV 오버레이
│   ├── metrics/
│   │   ├── latency_tracker.py     # 지연시간 P95/P99 측정
│   │   ├── accuracy_evaluator.py  # WER/CER 측정
│   │   └── metrics_store.py       # 메트릭 중앙 저장소
│   ├── dashboard/
│   │   ├── tui_dashboard.py       # Textual TUI 대시보드
│   │   └── web_dashboard.py       # FastAPI 웹 대시보드
│   └── config/
│       ├── schema.py              # Pydantic v2 AppConfig 스키마
│       └── config_manager.py      # YAML 로드, 환경변수 오버라이드, 핫스왑
└── tests/
    └── unit/                      # 단위 테스트
```

---

## 파이프라인 데이터 흐름

```
[FileMockCapture / DeckLinkCapture]
   │ audio_queue              │ video_queue
   ▼                           ▼
[AudioResampler]           [VideoCompositor] ◀──────────────────┐
   │                            │                      [SubtitleEvent]
   ▼                       [MetricsStore] ◀──────── [SubtitleManager]
[ClovaSpeechStreamer]            ▲                               ▲
   │ result_queue                │                               │
   └──────────────► [_stt_result_consumer] ─────────────────────┘

[MetricsStore] ──→ [TuiDashboard]
             └──→ [WebDashboard] (WebSocket)

[ConfigManager] ──watch()──→ Pipeline.apply_config()
                                  ├── AudioResampler.update_config()
                                  ├── SubtitleManager.update_config()
                                  └── VideoCompositor.update_style()
```

---

## 자막 내보내기

final 자막은 자동으로 `output/subtitles/` 에 SRT/VTT 형식으로 저장됩니다.

```
output/subtitles/subtitles.srt
output/subtitles/subtitles.vtt
```

수동으로 내보내려면:

```python
manager.export_srt("output/subtitles/session.srt")
manager.export_vtt("output/subtitles/session.vtt")
```

---

## 성공 기준

| 항목 | 목표 |
|------|------|
| End-to-end 자막 지연 | 2,000ms 이하 (목표 1,000ms) |
| 오디오 캡처 → STT 전송 지연 | 200ms 이하 |
| 프레임 드롭율 | 0.1% 이하 (1080p 30fps) |
| STT 연결 성공률 | 99% 이상 |
| 연속 운영 안정성 | 8시간 무중단 |

---

## 라이선스

Private
