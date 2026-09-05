#!/usr/bin/env python3
"""Generates a self-contained HTML dashboard from timing_output/<run>/ profiler CSVs.

Usage:
    python3 tools/timing_dashboard.py

Scans timing_output/ for run directories (each containing components_summary.csv,
optionally components_samples.csv and moves.csv, as written by
vision/pipeline/profiler.py), embeds all runs into one HTML file with a run
selector, and writes it to timing_output/dashboard.html. Open that file directly
in a browser -- no server needed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMING_DIR = REPO_ROOT / "timing_output"
OUTPUT_PATH = TIMING_DIR / "dashboard.html"


def parse_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["count"] = int(r["count"])
        for k in ("mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"):
            r[k] = float(r[k])
    return rows


def parse_moves(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if not r["latency_ms"]:
            continue  # incomplete row (run stopped mid-move)
        r["move_num"] = int(r["move_num"])
        r["latency_ms"] = float(r["latency_ms"])
        r["frames_to_detect"] = int(r["frames_to_detect"])
        out.append(r)
    return out


def parse_samples(path: Path) -> dict[str, list[float | None]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        data: dict[str, list[float | None]] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                v = row[c]
                data[c].append(float(v) if v not in (None, "") else None)
    return data


def collect_runs() -> dict:
    runs = {}
    for d in sorted(TIMING_DIR.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / "components_summary.csv"
        if not summary_path.exists():
            continue
        runs[d.name] = {
            "summary": parse_summary(summary_path),
            "moves": parse_moves(d / "moves.csv"),
            "samples": parse_samples(d / "components_samples.csv"),
        }
    order = sorted(runs.keys(), reverse=True)
    return {"runs": runs, "order": order}


TEMPLATE = r"""<meta charset="utf-8">
<title>Pipeline Telemetry</title>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f4f6f6;
    --surface-1: #ffffff;
    --surface-2: #eaeeee;
    --border: #d6dcdc;
    --text-primary: #12181a;
    --text-secondary: #4d5a5d;
    --text-muted: #7c898b;
    --accent: #2a78d6;
    --series-init: #4a3aa7;
    --series-hot: #2a78d6;
    --series-classify: #eb6834;
    --series-geom: #1baf7a;
    --series-frame: #256abf;
    --status-good: #0ca30c;
    --status-warning: #c98500;
    --grid-line: #dde3e3;
    --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #0f1416;
      --surface-1: #161d20;
      --surface-2: #1e2629;
      --border: #2b3538;
      --text-primary: #eef2f2;
      --text-secondary: #a8b4b6;
      --text-muted: #74807f;
      --accent: #3987e5;
      --series-init: #9085e9;
      --series-hot: #3987e5;
      --series-classify: #d95926;
      --series-geom: #199e70;
      --series-frame: #6da7ec;
      --status-good: #2fbf2f;
      --status-warning: #e0a334;
      --grid-line: #263033;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #0f1416;
    --surface-1: #161d20;
    --surface-2: #1e2629;
    --border: #2b3538;
    --text-primary: #eef2f2;
    --text-secondary: #a8b4b6;
    --text-muted: #74807f;
    --accent: #3987e5;
    --series-init: #9085e9;
    --series-hot: #3987e5;
    --series-classify: #d95926;
    --series-geom: #199e70;
    --series-frame: #6da7ec;
    --status-good: #2fbf2f;
    --status-warning: #e0a334;
    --grid-line: #263033;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--surface-0);
    color: var(--text-primary);
    font-family: var(--sans);
    padding: clamp(16px, 4vw, 44px);
    font-variant-numeric: tabular-nums;
  }
  .wrap { max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }

  header { display: flex; flex-direction: column; gap: 12px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
  .header-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  .eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); }
  h1 { font-size: clamp(24px, 3.2vw, 34px); margin: 0; font-weight: 650; text-wrap: balance; letter-spacing: -0.01em; }
  .subhead { color: var(--text-secondary); font-size: 14.5px; max-width: 62ch; line-height: 1.55; }

  .run-picker { display: flex; flex-direction: column; gap: 4px; align-items: flex-end; }
  .run-picker label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); }
  select#runSelect {
    font-family: var(--mono); font-size: 13px; color: var(--text-primary);
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 30px 8px 12px; cursor: pointer; min-width: 260px;
    appearance: none; -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237c898b'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 12px center;
  }
  select#runSelect:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  .stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .stat { background: var(--surface-1); padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
  .stat .label { font-size: 11.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .stat .value { font-family: var(--mono); font-size: 22px; font-weight: 600; }
  .stat .value small { font-size: 13px; font-weight: 500; color: var(--text-secondary); }

  section { display: flex; flex-direction: column; gap: 12px; }
  .section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  h2 { font-size: 16px; margin: 0; font-weight: 640; }
  .section-note { font-size: 12.5px; color: var(--text-muted); line-height: 1.5; max-width: 60ch; }

  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }
  .empty-note { color: var(--text-muted); font-size: 12.5px; font-family: var(--mono); padding: 10px 2px; }

  .legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--text-secondary); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; border-radius: 2.5px; display: inline-block; }

  .bar-row { display: grid; grid-template-columns: 152px 1fr 96px; align-items: center; gap: 10px; padding: 6px 0; font-size: 12.5px; }
  .bar-row .name { font-family: var(--mono); color: var(--text-secondary); font-size: 11.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-track { position: relative; height: 18px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
  .bar-fill { position: absolute; top: 0; left: 0; bottom: 0; border-radius: 4px 2px 2px 4px; }
  .bar-p95 { position: absolute; top: 2px; bottom: 2px; width: 2px; background: var(--text-primary); opacity: 0.35; }
  .bar-row .num { font-family: var(--mono); text-align: right; color: var(--text-primary); font-weight: 600; }
  .bar-row .num small { color: var(--text-muted); font-weight: 400; }
  .bar-row:hover .bar-track { outline: 2px solid var(--accent); outline-offset: 1px; }

  svg text { font-family: var(--mono); fill: var(--text-muted); }
  .axis-line { stroke: var(--grid-line); stroke-width: 1; }
  .grid-line { stroke: var(--grid-line); stroke-width: 1; stroke-dasharray: 2 3; }

  .trend-tooltip { position: absolute; pointer-events: none; background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 9px; font-family: var(--mono); font-size: 11.5px; line-height: 1.5; box-shadow: 0 4px 14px rgba(0,0,0,0.18); opacity: 0; transition: opacity 0.08s; white-space: nowrap; z-index: 5; }
  .trend-wrap { position: relative; }
  .crosshair { stroke: var(--text-muted); stroke-width: 1; opacity: 0; pointer-events: none; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  thead th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; padding: 0 10px 8px; border-bottom: 1px solid var(--border); }
  tbody td { padding: 7px 10px; border-bottom: 1px solid var(--surface-2); font-family: var(--mono); }
  tbody tr:last-child td { border-bottom: none; }
  .move-uci { font-weight: 600; color: var(--text-primary); }
  .mode-chip { display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px 2px 6px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .mode-exact { background: color-mix(in oklab, var(--status-good) 16%, transparent); color: var(--status-good); }
  .mode-fuzzy { background: color-mix(in oklab, var(--status-warning) 18%, transparent); color: var(--status-warning); }
  .mode-chip .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .latency-track { position: relative; height: 14px; background: var(--surface-2); border-radius: 3px; width: 140px; overflow: hidden; }
  .latency-fill { position: absolute; inset: 0; border-radius: 3px 2px 2px 3px; }
  .table-scroll { overflow-x: auto; }

  footer { color: var(--text-muted); font-size: 11.5px; font-family: var(--mono); border-top: 1px solid var(--border); padding-top: 14px; }
</style>

<div class="wrap">
  <header>
    <div class="header-top">
      <div style="display:flex;flex-direction:column;gap:6px;">
        <div class="eyebrow">chess-tracker &middot; vision pipeline profiler</div>
        <h1 id="pageTitle">Pipeline Telemetry</h1>
        <div class="subhead">Futásonkénti időzítési napló: inicializálás, kamera-frame-ek feldolgozása és a felismert lépések. A cél megtalálni, mely komponens viszi a legtöbb időt és hol vannak kiugró (lassú) frame-ek.</div>
      </div>
      <div class="run-picker">
        <label for="runSelect">Futás</label>
        <select id="runSelect"></select>
      </div>
    </div>
  </header>

  <div class="stat-strip" id="statStrip"></div>

  <section>
    <div class="section-head">
      <h2>Inicializálás</h2>
      <div class="section-note">Egyszeri költség a futás elején — a tábla első felismerése és a modellbetöltés.</div>
    </div>
    <div class="card" id="initBars"></div>
  </section>

  <section>
    <div class="section-head">
      <h2>Per-frame pipeline komponensek</h2>
      <div class="section-note">Minden kamera-frame-nél lefutó lépések átlaga (sáv) és p95-e (jelölő), logaritmikus skálán.</div>
    </div>
    <div class="card" id="hotBars"></div>
  </section>

  <section id="trendSection">
    <div class="section-head">
      <h2>frame_total és classifier_partial trend</h2>
      <div class="section-note">Minden mérési pont egy frame; vízszintes tengely a mérés sorszáma. A kiugrások gyakran fuzzy/bizonytalan felismerésekhez vagy újra-klasszifikációhoz köthetők.</div>
    </div>
    <div class="card trend-wrap">
      <div class="legend" style="margin-bottom:10px">
        <span><i class="swatch" style="background:var(--series-frame)"></i>frame_total</span>
        <span><i class="swatch" style="background:var(--series-classify)"></i>classifier_partial</span>
      </div>
      <div id="trendChart"></div>
      <div class="trend-tooltip" id="trendTip"></div>
    </div>
  </section>

  <section id="movesSection">
    <div class="section-head">
      <h2>Lépésenkénti felismerési idő</h2>
      <div class="section-note">Mennyi ideig tartott lépésenként a felismerés (<span style="font-family:var(--mono)">latency_ms</span>), hány frame kellett hozzá, és pontos (<span class="mode-chip mode-exact" style="padding:0 6px"><i class="dot"></i>exact</span>) vagy közelítő (<span class="mode-chip mode-fuzzy" style="padding:0 6px"><i class="dot"></i>fuzzy</span>) módszerrel sikerült.</div>
    </div>
    <div class="card table-scroll" id="movesCard">
      <table id="movesTable"></table>
    </div>
  </section>

  <footer id="footerNote">components_summary.csv &middot; components_samples.csv &middot; moves.csv</footer>
</div>

<script id="timing-data" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('timing-data').textContent);
  const cs = getComputedStyle(document.documentElement);
  const cvar = (n) => cs.getPropertyValue(n).trim();
  const fmt = (v, d = 1) => (v == null || Number.isNaN(v)) ? '–' : v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

  const runSelect = document.getElementById('runSelect');
  DATA.order.forEach((name) => {
    const run = DATA.runs[name];
    const frameRow = run.summary.find(r => r.component === 'frame_total');
    const frames = frameRow ? frameRow.count : 0;
    const moveCount = run.moves.length;
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name + '  (' + frames + ' frame' + (moveCount ? ', ' + moveCount + ' lépés' : '') + ')';
    runSelect.appendChild(opt);
  });
  runSelect.addEventListener('change', () => render(runSelect.value));

  function buildBars(container, byName, names, opts) {
    const rows = names.map(n => byName[n]).filter(Boolean);
    if (!rows.length) { container.innerHTML = '<div class="empty-note">Nincs adat ebben a futásban.</div>'; return; }
    const maxVal = opts.logScale
      ? Math.log10(Math.max(...rows.map(r => r.p95_ms), 1) + 1)
      : Math.max(...rows.map(r => r.p95_ms));
    const scale = (v) => {
      const x = opts.logScale ? Math.log10(v + 1) : v;
      return Math.max(2, maxVal > 0 ? (x / maxVal) * 100 : 2);
    };
    container.innerHTML = rows.map(r => {
      const color = opts.color(r.component);
      return `<div class="bar-row" title="${r.component}: mean ${fmt(r.mean_ms,2)}ms · p50 ${fmt(r.p50_ms,2)}ms · p95 ${fmt(r.p95_ms,2)}ms · n=${r.count}">
        <span class="name">${r.component}</span>
        <span class="bar-track">
          <span class="bar-fill" style="width:${scale(r.mean_ms)}%; background:${color}"></span>
          <span class="bar-p95" style="left:${scale(r.p95_ms)}%"></span>
        </span>
        <span class="num">${fmt(r.mean_ms, r.mean_ms < 10 ? 2 : 1)} <small>n=${r.count}</small></span>
      </div>`;
    }).join('');
  }

  function renderTrend(container, samples) {
    const ft = samples['frame_total_ms'] || [];
    const cp = samples['classifier_partial_ms'] || [];
    const n = ft.length;
    if (n < 2) { container.innerHTML = '<div class="empty-note">Nincs elég minta trendhez ebben a futásban.</div>'; return; }
    const W = 1080, H = 260, padL = 44, padR = 12, padT = 12, padB = 26;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const allVals = ft.filter(v => v != null).concat(cp.filter(v => v != null));
    const maxV = Math.max(...allVals, 1);
    const yScale = (v) => padT + plotH - (Math.log10(v + 1) / Math.log10(maxV + 1)) * plotH;
    const xScale = (i) => padL + (i / (n - 1)) * plotW;

    function pathFor(arr) {
      let d = '', started = false;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] == null) { started = false; continue; }
        const x = xScale(i), y = yScale(arr[i]);
        d += (started ? ' L ' : ' M ') + x.toFixed(1) + ' ' + y.toFixed(1);
        started = true;
      }
      return d;
    }

    const yTicks = [10, 100, 1000].filter(t => t <= maxV * 1.5);
    let gridSvg = yTicks.map(t => {
      const y = yScale(t);
      return `<line class="grid-line" x1="${padL}" x2="${W - padR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/>
              <text x="${padL - 8}" y="${(y + 3).toFixed(1)}" font-size="10" text-anchor="end">${t}</text>`;
    }).join('');

    const xTicks = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1];
    let xAxisSvg = xTicks.map(i => {
      const x = xScale(i);
      return `<text x="${x.toFixed(1)}" y="${H - 6}" font-size="10" text-anchor="middle">${i}</text>`;
    }).join('');

    const svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;overflow:visible" id="trendSvg">
      ${gridSvg}
      <line class="axis-line" x1="${padL}" x2="${padL}" y1="${padT}" y2="${H - padB}"/>
      <line class="axis-line" x1="${padL}" x2="${W - padR}" y1="${H - padB}" y2="${H - padB}"/>
      <path d="${pathFor(cp)}" fill="none" stroke="${cvar('--series-classify')}" stroke-width="1.4" opacity="0.85"/>
      <path d="${pathFor(ft)}" fill="none" stroke="${cvar('--series-frame')}" stroke-width="1.6"/>
      ${xAxisSvg}
      <line class="crosshair" id="crosshair" x1="0" x2="0" y1="${padT}" y2="${H - padB}"/>
      <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent" id="hoverRect" style="cursor:crosshair"/>
    </svg>`;
    container.innerHTML = svg;

    const hoverRect = document.getElementById('hoverRect');
    const crosshair = document.getElementById('crosshair');
    const tip = document.getElementById('trendTip');
    const svgEl = document.getElementById('trendSvg');
    tip.style.opacity = 0;

    hoverRect.addEventListener('mousemove', (e) => {
      const pt = svgEl.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const loc = pt.matrixTransform(svgEl.getScreenCTM().inverse());
      const i = Math.round(((loc.x - padL) / plotW) * (n - 1));
      if (i < 0 || i >= n) return;
      const x = xScale(i);
      crosshair.setAttribute('x1', x); crosshair.setAttribute('x2', x);
      crosshair.style.opacity = 1;
      const ftv = ft[i], cpv = cp[i];
      tip.innerHTML = `frame #${i}<br>frame_total: ${ftv != null ? fmt(ftv,1)+'ms' : '–'}<br>classifier_partial: ${cpv != null ? fmt(cpv,1)+'ms' : '–'}`;
      tip.style.opacity = 1;
      const wrapRect = svgEl.parentElement.getBoundingClientRect();
      let left = e.clientX - wrapRect.left + 14;
      if (left + 160 > wrapRect.width) left = e.clientX - wrapRect.left - 160;
      tip.style.left = left + 'px';
      tip.style.top = (e.clientY - wrapRect.top - 46) + 'px';
    });
    hoverRect.addEventListener('mouseleave', () => { tip.style.opacity = 0; crosshair.style.opacity = 0; });
  }

  function renderMoves(moves) {
    const section = document.getElementById('movesSection');
    if (!moves.length) { section.style.display = 'none'; return; }
    section.style.display = '';
    const maxLatency = Math.max(...moves.map(m => m.latency_ms));
    const theadHtml = `<thead><tr><th>#</th><th>Lépés</th><th>Frame-ek</th><th>Mód</th><th>Idő</th></tr></thead>`;
    const rowsHtml = moves.map(m => {
      const isFuzzy = m.mode.startsWith('fuzzy');
      const chip = isFuzzy
        ? `<span class="mode-chip mode-fuzzy"><i class="dot"></i>fuzzy</span> <span style="color:var(--text-muted)">${m.mode.replace('fuzzy ', '')}</span>`
        : `<span class="mode-chip mode-exact"><i class="dot"></i>exact</span>`;
      const pct = Math.max(2, (m.latency_ms / maxLatency) * 100);
      const color = isFuzzy ? cvar('--status-warning') : cvar('--series-frame');
      return `<tr>
        <td>${m.move_num}</td>
        <td class="move-uci">${m.uci}</td>
        <td>${m.frames_to_detect}</td>
        <td>${chip}</td>
        <td>
          <div style="display:flex;align-items:center;gap:8px">
            <span class="latency-track"><span class="latency-fill" style="width:${pct}%;background:${color}"></span></span>
            <span>${fmt(m.latency_ms / 1000, 2)}s</span>
          </div>
        </td>
      </tr>`;
    }).join('');
    document.getElementById('movesTable').innerHTML = theadHtml + '<tbody>' + rowsHtml + '</tbody>';
  }

  function render(runName) {
    const run = DATA.runs[runName];
    const summary = run.summary, moves = run.moves, samples = run.samples;
    const byName = Object.fromEntries(summary.map(r => [r.component, r]));

    document.getElementById('pageTitle').textContent = 'Pipeline Telemetry — ' + runName;
    document.getElementById('footerNote').textContent =
      'components_summary.csv · components_samples.csv · moves.csv — timing_output/' + runName;

    const frameRow = byName['frame_total'];
    const initRow = byName['init_total'];
    const totalMoveTime = moves.reduce((a, m) => a + m.latency_ms, 0);
    const fuzzyMoves = moves.filter(m => m.mode.startsWith('fuzzy')).length;
    const stats = [
      ['Init idő', initRow ? fmt(initRow.mean_ms, 0) : '–', 'ms'],
      ['Feldolgozott frame', frameRow ? frameRow.count.toLocaleString('en-US') : '0', ''],
      ['frame_total átlag', frameRow ? fmt(frameRow.mean_ms, 1) : '–', 'ms'],
      ['frame_total p95', frameRow ? fmt(frameRow.p95_ms, 1) : '–', 'ms'],
      ['Felismert lépés', moves.length.toLocaleString('en-US'), ''],
      ['Fuzzy lépés', moves.length ? (fuzzyMoves + ' / ' + moves.length) : '–', ''],
      ['Össz. lépésidő', moves.length ? fmt(totalMoveTime / 1000, 1) : '–', moves.length ? 's' : ''],
    ];
    document.getElementById('statStrip').innerHTML = stats.map(([label, value, unit]) =>
      `<div class="stat"><div class="label">${label}</div><div class="value">${value}${unit ? ` <small>${unit}</small>` : ''}</div></div>`
    ).join('');

    buildBars(document.getElementById('initBars'), byName,
      ['board_detect', 'model_load', 'init_total'],
      { logScale: false, color: () => cvar('--series-init') });

    const hotColor = (name) => {
      if (name.startsWith('classifier')) return cvar('--series-classify');
      if (name === 'frame_total') return cvar('--series-frame');
      if (['warp', 'square_diff', 'stabilizer'].includes(name)) return cvar('--series-geom');
      return cvar('--series-hot');
    };
    buildBars(document.getElementById('hotBars'), byName,
      ['frame_total', 'classifier_full', 'classifier_partial', 'resolve', 'warp', 'square_diff', 'apply_move', 'stabilizer'],
      { logScale: true, color: hotColor });

    renderTrend(document.getElementById('trendChart'), samples);
    renderMoves(moves);
  }

  runSelect.value = DATA.order[0];
  render(DATA.order[0]);
})();
</script>
"""


def main() -> None:
    data = collect_runs()
    if not data["order"]:
        raise SystemExit(f"Nem található egyetlen futás sem itt: {TIMING_DIR}")
    data_json = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA_JSON__", data_json)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard írva: {OUTPUT_PATH}  ({len(data['order'])} futás)")


if __name__ == "__main__":
    main()
