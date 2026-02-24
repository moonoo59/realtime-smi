"""
웹 대시보드 모듈입니다.

역할:
- FastAPI + WebSocket 기반 실시간 메트릭 스트리밍
- 브라우저에서 파이프라인 상태를 시각적으로 모니터링
- MetricsStore를 1초 주기로 폴링하여 클라이언트에게 브로드캐스트

엔드포인트:
    GET  /              HTML 대시보드 페이지
    GET  /api/snapshot  현재 메트릭 스냅샷 (JSON)
    GET  /api/health    헬스체크
    WS   /ws/metrics    실시간 메트릭 스트림
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Optional, Set

logger = logging.getLogger(__name__)


# =========================================================================
# 알림 정의
# =========================================================================

ALERT_RULES = {
    "STT_DISCONNECTED": "STT 서버 연결 끊김",
    "HIGH_DROP_RATE": "프레임 드롭률 과다 (>5%)",
    "HIGH_E2E_LATENCY": "E2E 지연시간 초과 (>3초)",
    "AUDIO_SILENCE": "오디오 무음 감지 (RMS<0.001)",
    "STT_ERROR_BURST": "STT 오류 급증 (>10회)",
}

ALERT_DEDUP_SEC = 10  # 동일 알림 재발생 억제 시간(초)


# =========================================================================
# HTML 템플릿 (인라인)
# =========================================================================

_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SDI Realtime Subtitle — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --partial-color: #a8b5c3;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 16px; font-weight: 600; letter-spacing: 0.5px; }
  #conn-status {
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 12px;
    background: var(--border);
  }
  #conn-status.connected { background: #1a3a1f; color: var(--green); }
  #conn-status.disconnected { background: #3a1a1a; color: var(--red); }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    padding: 14px;
  }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .panel-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 12px;
  }

  /* 자막 패널 */
  #subtitle-text {
    font-size: 22px;
    font-weight: 500;
    min-height: 60px;
    word-break: break-all;
    line-height: 1.4;
  }
  #subtitle-text.partial { color: var(--partial-color); }
  #subtitle-text.final { color: var(--text); }

  /* 오디오 레벨 */
  .meter-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .meter-label { width: 40px; font-size: 12px; color: var(--text-muted); }
  .meter-bar {
    flex: 1;
    height: 12px;
    background: var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .meter-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.15s ease;
    background: linear-gradient(90deg, var(--green) 0%, var(--yellow) 70%, var(--red) 100%);
  }
  .meter-value { width: 50px; font-size: 12px; text-align: right; }

  /* STT 상태 */
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat-item { }
  .stat-value { font-size: 22px; font-weight: 600; }
  .stat-label { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
  }
  .badge.ok { background: #1a3a1f; color: var(--green); }
  .badge.err { background: #3a1a1a; color: var(--red); }

  /* 프레임 통계 */
  .kv-list { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .kv-item { }
  .kv-val { font-size: 18px; font-weight: 600; }
  .kv-key { font-size: 11px; color: var(--text-muted); }

  /* 지연시간 차트 */
  #latency-chart-wrap { position: relative; height: 160px; }

  /* WER/CER */
  .accuracy-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }

  /* 파이프라인 제어 패널 */
  .ctrl-panel select,
  .ctrl-panel input[type=text] {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 13px;
  }
  .ctrl-panel button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 5px 14px;
    cursor: pointer;
    font-size: 13px;
  }
  .ctrl-panel button:hover { opacity: 0.85; }
  #ctrl-stop { background: var(--red); }
  #ctrl-status { font-size: 13px; color: var(--text-muted); }

  /* 알림 배너 */
  #alerts {
    position: fixed;
    bottom: 16px;
    right: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 999;
    max-width: 340px;
  }
  .alert-item {
    background: #3a2a0a;
    border: 1px solid var(--yellow);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    animation: slideIn 0.2s ease;
  }
  .alert-item.error { background: #3a1a1a; border-color: var(--red); }
  @keyframes slideIn {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
  }
</style>
</head>
<body>
<header>
  <h1>SDI Realtime Subtitle Dashboard</h1>
  <span id="conn-status" class="disconnected">연결 중...</span>
</header>

<div class="grid">
  <!-- 파이프라인 제어 패널 -->
  <div class="panel ctrl-panel" style="grid-column: 1 / -1; margin-bottom: 12px;">
    <div class="panel-title">파이프라인 제어</div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">
      <select id="ctrl-mode">
        <option value="file">파일 모드</option>
        <option value="live">라이브 모드</option>
      </select>
      <input id="ctrl-audio" type="text" placeholder="오디오/영상 파일 경로 (예: /Users/.../video.avi)" style="flex:1; min-width:300px;">
      <label><input type="checkbox" id="ctrl-no-stt"> STT 비활성화</label>
    </div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <input id="ctrl-secret-key" type="password" placeholder="Clova Secret Key (STT 사용 시 입력)" style="flex:1; min-width:260px;">
      <button id="ctrl-start" onclick="startPipeline()">▶ 시작</button>
      <button id="ctrl-stop" onclick="stopPipeline()">■ 중지</button>
      <span id="ctrl-status">대기 중</span>
    </div>
  </div>

  <!-- 패널 1: 현재 자막 -->
  <div class="panel" style="grid-column: 1 / -1;">
    <div class="panel-title">현재 자막</div>
    <div id="subtitle-text" class="partial">—</div>
  </div>

  <!-- 패널 2: 오디오 레벨 -->
  <div class="panel">
    <div class="panel-title">오디오 레벨</div>
    <div class="meter-row">
      <span class="meter-label">RMS</span>
      <div class="meter-bar"><div class="meter-fill" id="rms-bar" style="width:0%"></div></div>
      <span class="meter-value" id="rms-val">0.000</span>
    </div>
    <div class="meter-row">
      <span class="meter-label">Peak</span>
      <div class="meter-bar"><div class="meter-fill" id="peak-bar" style="width:0%"></div></div>
      <span class="meter-value" id="peak-val">0.000</span>
    </div>
  </div>

  <!-- 패널 3: STT 상태 -->
  <div class="panel">
    <div class="panel-title">STT 상태</div>
    <div class="stat-grid">
      <div class="stat-item">
        <div><span id="stt-conn" class="badge err">OFFLINE</span></div>
        <div class="stat-label">연결 상태</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="stt-errors">0</div>
        <div class="stat-label">오류 횟수</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="stt-reconnects">0</div>
        <div class="stat-label">재연결 횟수</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="stt-last">—</div>
        <div class="stat-label">마지막 결과</div>
      </div>
    </div>
  </div>

  <!-- 패널 4: 지연시간 -->
  <div class="panel">
    <div class="panel-title">지연시간 (ms)</div>
    <div id="latency-chart-wrap">
      <canvas id="latency-chart"></canvas>
    </div>
  </div>

  <!-- 패널 5: 프레임 통계 -->
  <div class="panel">
    <div class="panel-title">프레임 통계</div>
    <div class="kv-list">
      <div class="kv-item">
        <div class="kv-val" id="f-total">0</div>
        <div class="kv-key">총 프레임</div>
      </div>
      <div class="kv-item">
        <div class="kv-val" id="f-drop">0</div>
        <div class="kv-key">드롭 수</div>
      </div>
      <div class="kv-item">
        <div class="kv-val" id="f-rate">0.0%</div>
        <div class="kv-key">드롭률</div>
      </div>
      <div class="kv-item">
        <div class="kv-val" id="f-queue">0</div>
        <div class="kv-key">큐 깊이</div>
      </div>
    </div>
  </div>

  <!-- 패널 6: WER/CER -->
  <div class="panel">
    <div class="panel-title">인식 정확도</div>
    <div class="accuracy-grid">
      <div class="stat-item">
        <div class="stat-value" id="acc-wer">—</div>
        <div class="stat-label">WER</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="acc-cer">—</div>
        <div class="stat-label">CER</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="acc-pairs">0</div>
        <div class="stat-label">평가 쌍</div>
      </div>
    </div>
  </div>
</div>

<div id="alerts"></div>

<script>
// =========================================================================
// Chart.js 지연시간 라인 차트
// =========================================================================
const MAX_POINTS = 60;
const latencyLabels = [];
const e2eData = [];
const partialData = [];

const ctx = document.getElementById('latency-chart').getContext('2d');
const latencyChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: latencyLabels,
    datasets: [
      {
        label: 'E2E',
        data: e2eData,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      },
      {
        label: 'Partial',
        data: partialData,
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
    scales: {
      x: { display: false },
      y: {
        ticks: { color: '#8b949e', font: { size: 11 } },
        grid: { color: '#21262d' },
        beginAtZero: true,
      },
    },
  },
});

function pushLatency(e2eMs, partialMs) {
  const label = new Date().toLocaleTimeString();
  latencyLabels.push(label);
  e2eData.push(e2eMs);
  partialData.push(partialMs);
  if (latencyLabels.length > MAX_POINTS) {
    latencyLabels.shift(); e2eData.shift(); partialData.shift();
  }
  latencyChart.update('none');
}

// =========================================================================
// 알림 표시 (10초 자동 제거)
// =========================================================================
const activeAlerts = {};
function showAlert(key, msg, isError = false) {
  if (activeAlerts[key]) return;
  const el = document.createElement('div');
  el.className = 'alert-item' + (isError ? ' error' : '');
  el.textContent = msg;
  document.getElementById('alerts').appendChild(el);
  activeAlerts[key] = true;
  setTimeout(() => {
    el.remove();
    delete activeAlerts[key];
  }, 10000);
}

// =========================================================================
// UI 업데이트 헬퍼
// =========================================================================
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function setWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min(100, pct * 100).toFixed(1) + '%';
}
function relativeTime(ns) {
  if (!ns) return '—';
  const sec = (Date.now() - ns / 1e6) / 1000;
  if (sec < 2) return '방금';
  if (sec < 60) return Math.floor(sec) + '초 전';
  return Math.floor(sec / 60) + '분 전';
}

// =========================================================================
// 메트릭 렌더링
// =========================================================================
function render(data) {
  // 자막
  const sub = data.subtitle || {};
  const subEl = document.getElementById('subtitle-text');
  subEl.textContent = sub.text || '—';
  subEl.className = sub.is_partial ? 'partial' : 'final';

  // 오디오
  const audio = data.audio || {};
  setWidth('rms-bar', audio.rms || 0);
  setWidth('peak-bar', audio.peak || 0);
  setText('rms-val', (audio.rms || 0).toFixed(3));
  setText('peak-val', (audio.peak || 0).toFixed(3));

  // STT
  const stt = data.stt || {};
  const connEl = document.getElementById('stt-conn');
  if (stt.connected) {
    connEl.textContent = 'ONLINE';
    connEl.className = 'badge ok';
  } else {
    connEl.textContent = 'OFFLINE';
    connEl.className = 'badge err';
  }
  setText('stt-errors', stt.error_count ?? 0);
  setText('stt-reconnects', stt.reconnect_count ?? 0);
  setText('stt-last', relativeTime(stt.last_result_at_ns));

  // 프레임
  const frames = data.frames || {};
  setText('f-total', frames.total_frames ?? 0);
  setText('f-drop', frames.drop_count ?? 0);
  setText('f-rate', ((frames.drop_rate || 0) * 100).toFixed(1) + '%');
  setText('f-queue', frames.queue_depth ?? 0);

  // 지연시간 차트
  const latency = data.latency || {};
  const e2e = latency.e2e || {};
  const partial = latency.partial || {};
  pushLatency(
    Math.round((e2e.p95_ms || 0)),
    Math.round((partial.p95_ms || 0))
  );

  // 정확도
  const acc = data.accuracy || {};
  setText('acc-wer', acc.pair_count > 0 ? (acc.wer * 100).toFixed(1) + '%' : '—');
  setText('acc-cer', acc.pair_count > 0 ? (acc.cer * 100).toFixed(1) + '%' : '—');
  setText('acc-pairs', acc.pair_count ?? 0);

  // 파이프라인 상태
  if (data.pipeline_status !== undefined) {
    document.getElementById('ctrl-status').textContent =
      pipelineStatusMap[data.pipeline_status] || data.pipeline_status;
  }

  // 알림 체크
  const alerts = data.alerts || [];
  alerts.forEach(a => showAlert(a.key, a.message, a.level === 'error'));
}

// =========================================================================
// 파이프라인 제어
// =========================================================================
const pipelineStatusMap = {
  'running': '실행 중',
  'idle': '대기 중',
  'stopping': '종료 중',
  'error': '오류',
  'unavailable': '사용 불가',
};

async function startPipeline() {
  const body = {
    mode: document.getElementById('ctrl-mode').value,
    audio_path: document.getElementById('ctrl-audio').value,
    no_stt: document.getElementById('ctrl-no-stt').checked,
    secret_key: document.getElementById('ctrl-secret-key').value,
  };
  const res = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json();
  document.getElementById('ctrl-status').textContent = data.message;
}

async function stopPipeline() {
  const res = await fetch('/api/stop', { method: 'POST' });
  const data = await res.json();
  document.getElementById('ctrl-status').textContent = data.message;
}

// 페이지 로드 시 파이프라인 초기 설정을 폼에 반영
async function loadInitialConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    if (cfg.no_stt) document.getElementById('ctrl-no-stt').checked = true;
    const modeEl = document.getElementById('ctrl-mode');
    if (cfg.mode) modeEl.value = cfg.mode;
  } catch (e) { /* 무시 */ }
}
loadInitialConfig();

// =========================================================================
// WebSocket 연결 (자동 재연결)
// =========================================================================
const statusEl = document.getElementById('conn-status');
let ws = null;
let reconnectTimer = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws/metrics');

  ws.onopen = () => {
    statusEl.textContent = '연결됨';
    statusEl.className = 'connected';
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      render(data);
    } catch (e) { /* ignore parse error */ }
  };

  ws.onclose = () => {
    statusEl.textContent = '연결 끊김 — 재연결 중...';
    statusEl.className = 'disconnected';
    reconnectTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => { ws.close(); };
}

connect();
</script>
</body>
</html>"""


# =========================================================================
# WebDashboard 클래스
# =========================================================================

class WebDashboard:
    """
    FastAPI 기반 웹 대시보드입니다.

    MetricsStore를 1초 주기로 폴링하여 WebSocket 클라이언트에게 브로드캐스트합니다.
    """

    def __init__(self, metrics_store, pipeline=None, host: str = "0.0.0.0", port: int = 8765, config=None) -> None:
        self._store = metrics_store
        self._pipeline = pipeline
        self._config = config
        self._host = host
        self._port = port
        self._clients: Set = set()
        self._app = None
        self._server_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._running = False
        # 알림 중복 방지: {key: last_fired_time_sec}
        self._alert_last: dict[str, float] = {}

        self._build_app()

    def _build_app(self) -> None:
        """FastAPI 앱과 라우트를 구성합니다."""
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse, JSONResponse
        except ImportError:
            raise RuntimeError(
                "fastapi와 uvicorn이 설치되어 있지 않습니다. "
                "pip install fastapi uvicorn 을 실행하세요."
            )

        try:
            from fastapi import Request as _Request
        except ImportError:
            _Request = None  # type: ignore[assignment,misc]

        # from __future__ import annotations 환경에서 FastAPI가 타입을
        # 문자열로 평가할 때 모듈 글로벌 네임스페이스에서 찾을 수 있도록 등록
        import sys as _sys
        _mod = _sys.modules[__name__]
        if not hasattr(_mod, "WebSocket"):
            _mod.WebSocket = WebSocket  # type: ignore[attr-defined]
        if not hasattr(_mod, "WebSocketDisconnect"):
            _mod.WebSocketDisconnect = WebSocketDisconnect  # type: ignore[attr-defined]
        if _Request is not None and not hasattr(_mod, "Request"):
            _mod.Request = _Request  # type: ignore[attr-defined]

        app = FastAPI(title="SDI Realtime Subtitle Dashboard", docs_url=None, redoc_url=None)

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(content=_HTML)

        @app.get("/api/snapshot")
        async def snapshot():
            return JSONResponse(content=self._collect_metrics())

        @app.get("/api/health")
        async def health():
            return JSONResponse(content={"status": "ok", "ts": time.time()})

        @app.get("/api/status")
        async def pipeline_status():
            if self._pipeline is None:
                return JSONResponse(content={"status": "unavailable"})
            return JSONResponse(content={"status": self._pipeline.get_status()})

        @app.post("/api/start")
        async def start_pipeline(request: Request):
            if self._pipeline is None:
                return JSONResponse(status_code=400, content={"message": "파이프라인이 없습니다."})
            if self._pipeline.get_status() == "running":
                return JSONResponse(status_code=400, content={"message": "파이프라인이 이미 실행 중입니다."})

            body = await request.json()
            audio_path = body.get("audio_path", "")
            video_path = body.get("video_path", "")
            mode = body.get("mode", "file")
            no_stt = body.get("no_stt", False)
            secret_key = body.get("secret_key", "")

            # 파이프라인 설정 업데이트 (오디오/비디오 경로, 모드, Secret Key)
            if self._pipeline._config is not None:
                from src.config.schema import AppConfig
                config_dict = self._pipeline._config.model_dump()
                config_dict["system"]["mode"] = mode
                if audio_path:
                    config_dict["capture"]["test_file"]["audio_path"] = audio_path
                if video_path:
                    config_dict["capture"]["test_file"]["video_path"] = video_path
                if secret_key:
                    config_dict["stt"]["secret_key"] = secret_key
                self._pipeline._config = AppConfig(**config_dict)

            self._pipeline._no_stt = no_stt

            # 백그라운드 태스크로 파이프라인 실행
            self._pipeline_task = asyncio.create_task(self._pipeline.run())
            return JSONResponse(content={"message": "파이프라인 시작됨"})

        @app.post("/api/stop")
        async def stop_pipeline():
            if self._pipeline is None:
                return JSONResponse(status_code=400, content={"message": "파이프라인이 없습니다."})
            self._pipeline.request_shutdown()
            return JSONResponse(content={"message": "파이프라인 종료 요청됨"})

        @app.get("/api/config")
        async def get_config():
            """현재 파이프라인 초기 설정을 반환합니다 (UI 폼 초기화용)."""
            if self._pipeline is None:
                return JSONResponse(content={"no_stt": False, "mode": "file"})
            cfg = self._pipeline._config
            return JSONResponse(content={
                "no_stt": self._pipeline._no_stt,
                "mode": cfg.system.mode if cfg else "file",
            })

        @app.websocket("/ws/metrics")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)
            logger.debug("WebSocket 클라이언트 연결: %s", websocket.client)
            try:
                while True:
                    # 클라이언트로부터 메시지를 받아도 무시 (ping 방지용 루프)
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    except asyncio.TimeoutError:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)
                logger.debug("WebSocket 클라이언트 연결 종료")

        self._app = app

    def _collect_metrics(self) -> dict:
        """MetricsStore에서 현재 메트릭을 수집하여 dict로 반환합니다."""
        audio = self._store.get_audio_level()
        stt = self._store.get_stt_status()
        frames = self._store.get_frame_stats()
        subtitle = self._store.get_current_subtitle()
        accuracy = self._store.get_accuracy_stats()
        latency_all = self._store.get_latency_stats()

        # 지연시간: stage별 p95_ms 추출
        latency = {}
        for stage, stats in (latency_all or {}).items():
            if stats is not None and hasattr(stats, "p95_ms"):
                latency[stage] = {"p95_ms": stats.p95_ms, "p99_ms": getattr(stats, "p99_ms", 0)}

        alerts = self._check_alerts(audio, stt, frames, latency)

        return {
            "ts": time.time(),
            "pipeline_status": self._pipeline.get_status() if self._pipeline is not None else "unavailable",
            "subtitle": {
                "text": subtitle.text,
                "is_partial": subtitle.is_partial,
            },
            "audio": {
                "rms": round(audio.rms, 4),
                "peak": round(audio.peak, 4),
            },
            "stt": {
                "connected": stt.connected,
                "error_count": stt.error_count,
                "reconnect_count": stt.reconnect_count,
                "last_result_at_ns": stt.last_result_at_ns,
            },
            "frames": {
                "total_frames": frames.total_frames,
                "drop_count": frames.drop_count,
                "drop_rate": round(frames.drop_rate, 4),
                "queue_depth": frames.queue_depth,
            },
            "latency": latency,
            "accuracy": {
                "wer": round(accuracy.wer, 4),
                "cer": round(accuracy.cer, 4),
                "pair_count": accuracy.pair_count,
            },
            "alerts": alerts,
        }

    def _check_alerts(self, audio, stt, frames, latency) -> list[dict]:
        """현재 메트릭을 기반으로 발생한 알림 목록을 반환합니다 (10초 중복 억제)."""
        now = time.time()
        alerts = []

        def maybe_alert(key: str, message: str, level: str = "warning"):
            last = self._alert_last.get(key, 0)
            if now - last >= ALERT_DEDUP_SEC:
                self._alert_last[key] = now
                alerts.append({"key": key, "message": message, "level": level})

        if not stt.connected:
            maybe_alert("STT_DISCONNECTED", "STT 서버 연결이 끊어졌습니다.", "error")

        if frames.drop_rate > 0.05:
            maybe_alert(
                "HIGH_DROP_RATE",
                f"프레임 드롭률 {frames.drop_rate*100:.1f}% 초과",
            )

        e2e = latency.get("e2e", {})
        if e2e.get("p95_ms", 0) > 3000:
            maybe_alert(
                "HIGH_E2E_LATENCY",
                f"E2E 지연 P95={e2e['p95_ms']:.0f}ms",
                "error",
            )

        if audio.rms < 0.001 and audio.updated_at_ns > 0:
            maybe_alert("AUDIO_SILENCE", "오디오 무음이 감지되었습니다.")

        if stt.error_count > 10:
            maybe_alert(
                "STT_ERROR_BURST",
                f"STT 오류 {stt.error_count}회 발생",
                "error",
            )

        return alerts

    async def _broadcast_loop(self) -> None:
        """1초 주기로 메트릭을 수집하여 모든 WebSocket 클라이언트에게 브로드캐스트합니다."""
        while self._running:
            try:
                if self._clients:
                    payload = json.dumps(self._collect_metrics())
                    dead = set()
                    for client in list(self._clients):
                        try:
                            await client.send_text(payload)
                        except Exception:
                            dead.add(client)
                    self._clients -= dead
            except Exception as exc:
                logger.warning("브로드캐스트 오류: %s", exc)
            await asyncio.sleep(1.0)

    async def start(self) -> None:
        """대시보드 서버와 브로드캐스트 루프를 시작합니다."""
        try:
            import uvicorn
        except ImportError:
            raise RuntimeError("uvicorn이 설치되지 않았습니다. pip install uvicorn 을 실행하세요.")

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())

        logger.info(
            "웹 대시보드 시작: http://%s:%d",
            "localhost" if self._host == "0.0.0.0" else self._host,
            self._port,
        )

    async def stop(self) -> None:
        """대시보드를 종료합니다."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        logger.info("웹 대시보드 종료")

    def update_accuracy(self, wer: float, cer: float, pair_count: int) -> None:
        """AccuracyEvaluator 결과를 MetricsStore에 기록합니다."""
        self._store.update_accuracy_stats(wer=wer, cer=cer, pair_count=pair_count)
