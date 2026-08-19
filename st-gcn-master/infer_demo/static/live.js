/* ============================================================
   Live 实时检测 - 累积 300 帧 + ST-GCN/KNN 滑动窗口
   ============================================================ */

const $ = (id) => document.getElementById(id);

const els = {
  video: $('live-video'),
  canvas: $('live-canvas'),
  wrap: $('live-video-wrap'),
  file: $('live-file'),
  fileDrop: $('file-drop'),
  fileName: $('file-name'),
  fileZone: $('file-zone'),
  path: $('live-path'),
  status: $('live-status'),
  phaseBar: $('live-phase-bar'),
  phaseFill: $('phase-fill'),
  phaseDetail: $('phase-detail'),
  frameCount: $('live-frame-count'),
  fps: $('live-fps'),
  nextInfer: $('live-next-infer'),
  reset: $('live-reset'),
  interval: $('live-interval'),
  conf: $('live-conf'),
  method: $('live-method'),
  overlayTop: $('live-overlay-top'),
  fpsBadge: $('live-fps-badge'),
  phaseBadge: $('live-phase-badge'),
  empty: $('live-empty'),
  warming: $('live-warming'),
  warmingProgress: $('warming-progress'),
  warmingBarFill: $('warming-bar-fill'),
  startOverlay: $('live-start'),
  resultCard: $('live-result-card'),
  result: $('live-result'),
  confFill: $('live-conf-fill'),
  warmingError: $('warming-error'),
  warmingErrorDetail: $('warming-error-detail'),
  warmingReplay: $('warming-replay'),
  tabs: document.querySelectorAll('.tab'),
  tabPanels: { upload: $('tab-upload'), path: $('tab-path') },
};

const EDGES = [
  [0, 1], [0, 14], [0, 15], [14, 16], [15, 17],
  [1, 2], [2, 3], [3, 4],
  [1, 5], [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10],
  [1, 11], [11, 12], [12, 13],
];

let loopTimer = null;
let inflight = false;
let fpsAvg = 0;
let frameIdx = 0;
let videoUrl = null;
let _startBound = false;

init();

function init() {
  bindTabs();
  bindFile();
  bindPath();
  bindReset();
  bindResize();
  bindReplay();
}

function bindReplay() {
  els.warmingReplay.addEventListener('click', async () => {
    console.log('[live] user clicked 重试自动播放');
    try {
      await els.video.play();
      hideVideoError();
      setStatus('重试成功');
    } catch (e) {
      console.warn('[live] replay play() failed:', e);
      setStatus('还是不行，请直接点视频自带的 ▶');
    }
  });
}

function showVideoError(title, detail) {
  if (!els.warmingError) return;
  els.warmingError.classList.remove('hidden');
  if (els.warmingErrorDetail) {
    els.warmingErrorDetail.textContent = `${title} — ${detail}（readyState=${els.video.readyState}, videoWidth=${els.video.videoWidth}, paused=${els.video.paused}）`;
  }
}

function hideVideoError() {
  if (!els.warmingError) return;
  els.warmingError.classList.add('hidden');
}

function bindTabs() {
  els.tabs.forEach(t => {
    t.addEventListener('click', () => {
      els.tabs.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      Object.values(els.tabPanels).forEach(p => p.classList.add('hidden'));
      els.tabPanels[t.dataset.tab].classList.remove('hidden');
    });
  });
}
function bindFile() {
  els.file.addEventListener('change', e => {
    if (e.target.files[0]) loadVideoFromFile(e.target.files[0]);
  });

  // 拖拽上传
  const zone = els.fileDrop;
  if (zone) {
    ['dragenter', 'dragover'].forEach(ev =>
      zone.addEventListener(ev, e => {
        e.preventDefault(); e.stopPropagation();
        zone.classList.add('drag');
      }));
    ['dragleave', 'dragend'].forEach(ev =>
      zone.addEventListener(ev, e => {
        e.preventDefault(); e.stopPropagation();
        zone.classList.remove('drag');
      }));
    zone.addEventListener('drop', e => {
      e.preventDefault(); e.stopPropagation();
      zone.classList.remove('drag');
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) {
        // 同步到隐藏 input，保证原生 change 也能感知（便于后面重置）
        try { const dt = new DataTransfer(); dt.items.add(f); els.file.files = dt.files; } catch (_) {}
        loadVideoFromFile(f);
      }
    });
  }

  // “✕” 重新选择：清空 input 与文件名显示
  if (els.fileName) {
    const clear = els.fileName.querySelector('.fn-clear');
    if (clear) clear.addEventListener('click', e => {
      e.preventDefault(); e.stopPropagation();
      els.file.value = '';
      els.fileName.hidden = true;
    });
  }
}

function showFileName(name) {
  if (!els.fileName) return;
  els.fileName.hidden = false;
  const t = els.fileName.querySelector('.fn-text');
  if (t) t.textContent = name;
}
function bindPath() {
  els.path.addEventListener('change', () => {
    if (els.path.value) loadVideoFromUrl(els.path.value);
  });
}
function bindReset() {
  els.reset.addEventListener('click', async () => {
    await fetch('/api/live_reset', { method: 'POST' });
    setStatus('缓冲已重置');
    setWarming(0, 300);
    setResult('—', 0);
    els.warming.classList.remove('hidden');
    frameIdx = 0;
    els.resultCard.classList.add('hidden');
    // 重新进入 warming 显示（如果 ready 时切换过）
    if (els.warmingBarFill) els.warmingBarFill.style.width = '0%';
  });
}
function bindResize() {
  const sync = () => {
    const r = els.video.getBoundingClientRect();
    els.canvas.width = r.width;
    els.canvas.height = r.height;
  };
  new ResizeObserver(sync).observe(els.wrap);
  els.video.addEventListener('loadedmetadata', sync);
}

function loadVideoFromFile(file) {
  console.log('[live] file selected:', file.name, 'type=' + file.type,
              'size=' + file.size, 'lastModified=' + file.lastModified);
  // 立刻看 MIME 类型——如果不对，立即警告
  if (file.type && !file.type.startsWith('video/')) {
    console.warn('[live] ⚠️ 文件 MIME 不是 video/*:', file.type);
    setStatus('⚠️ 文件类型 ' + file.type + ' 不是视频！请选真正的 mp4 文件');
  }
  if (file.size === 0) {
    setStatus('⚠️ 文件是空的（0 字节）！');
    return;
  }
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  els.video.src = videoUrl;
  els.video.muted = true;
  els.video.loop = true;
  els.video.load();                          // 显式触发加载
  els.empty.classList.add('hidden');
  els.startOverlay.classList.remove('hidden');
  bindStartOverlay();
  showFileName(file.name);
  setStatus('视频已加载 (' + (file.type || '未知类型') + ', ' +
            (file.size / 1024 / 1024).toFixed(2) + ' MB)，点击 ▶ 开始');
  els.video.onloadedmetadata = () => {
    console.log('[live] video metadata OK:', els.video.videoWidth, 'x', els.video.videoHeight,
                'readyState=' + els.video.readyState);
  };
  // 监听加载错误
  els.video.onerror = () => {
    const err = els.video.error;
    console.error('[live] video.onerror:', err ? err.code : 'null',
                  err ? ({1:'ABORTED',2:'NETWORK',3:'DECODE',4:'SRC_NOT_SUPPORTED'}[err.code] || '?') : '');
    setStatus('⚠️ 视频加载失败（error.code=' + (err ? err.code : 'null') + '）' +
              '——可能是格式不支持，请换 mp4 (H.264 Baseline)');
  };
  els.video.onended = () => els.video.play();
}

function loadVideoFromUrl(_u) {
  els.video.src = els.path.value;
  els.video.muted = true;
  els.video.loop = true;
  els.video.load();
  els.empty.classList.add('hidden');
  els.startOverlay.classList.remove('hidden');
  bindStartOverlay();
  setStatus('服务器视频已加载，点击 ▶ 开始');
  els.video.onloadedmetadata = () => {
    console.log('[live] video metadata:', els.video.videoWidth, 'x', els.video.videoHeight,
                'readyState=' + els.video.readyState);
  };
  els.video.onerror = () => {
    const err = els.video.error;
    console.error('[live] video.onerror:', err ? err.code : 'null');
    setStatus('⚠️ 视频加载失败（error.code=' + (err ? err.code : 'null') + '）——请换 mp4 (H.264)');
  };
  els.video.onended = () => els.video.play();
}

function bindStartOverlay() {
  if (_startBound) return;
  _startBound = true;

  // 用 addEventListener 比 onclick 更稳（防止被框架/扩展替换）
  const onStart = (e) => {
    console.log('[live] start triggered', e && e.type);
    doStart();
  };
  els.startOverlay.addEventListener('click', onStart);
  els.startOverlay.addEventListener('pointerdown', onStart);

  // 兜底：点视频区本身（包括 controls 附近的死区）也能启动
  els.video.addEventListener('click', (e) => {
    if (els.video.paused) { doStart(); e.preventDefault(); }
  });

  // 兜底：整个 wrap 也能启动（除非用户点的目标是真正的控件）
  els.wrap.addEventListener('click', (e) => {
    // 如果 start overlay 已经隐藏说明已启动，不再重触
    if (els.startOverlay.classList.contains('hidden')) return;
    // 如果点的是真的控件（button / select / input / control），别动
    const tag = (e.target.tagName || '').toUpperCase();
    if (['BUTTON', 'INPUT', 'SELECT', 'A', 'LABEL'].includes(tag)) return;
    doStart();
  });

  // 键盘兜底：按 空格 / 回车 / S 键 也能启动
  const onKey = (e) => {
    if (els.startOverlay.classList.contains('hidden')) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if ([' ', 'Enter', 's', 'S'].includes(e.key)) {
      e.preventDefault();
      doStart();
    }
  };
  document.addEventListener('keydown', onKey);
}

function doStart() {
  els.startOverlay.classList.add('hidden');
  els.warming.classList.remove('hidden');
  setWarming(0, 300);
  setStatus('正在启动...');

  // 强制 muted/autoplay/loop，确保 Chrome 自动播放策略通过
  els.video.muted = true;
  els.video.loop = true;
  els.video.playsInline = true;
  els.video.autoplay = true;

  const tryPlay = () => {
    const p = els.video.play();
    if (p && p.catch) {
      p.catch(err => {
        console.warn('[live] play() rejected:', err);
        setStatus('浏览器拒绝自动播放，请点视频自带控制条的 ▶');
        els.startOverlay.classList.remove('hidden');
        els.warming.classList.add('hidden');
      });
    }
  };
  if (els.video.readyState >= 2 && els.video.videoWidth > 0) {
    console.log('[live] readyState=2 videoWidth=' + els.video.videoWidth + ', try play');
    tryPlay();
  } else {
    console.log('[live] waiting canplay (readyState=' + els.video.readyState +
                ' videoWidth=' + els.video.videoWidth + ')');
    els.video.oncanplay = () => {
      console.log('[live] canplay fired, try play');
      tryPlay();
      els.video.oncanplay = null;
    };
  }

  startLoop();

  // 兜底：5 秒后如果还 paused 且没有 videoWidth，提示用户
  setTimeout(() => {
    if (els.video.paused || !els.video.videoWidth) {
      console.warn('[live] 5s 后仍 paused=', els.video.paused,
                  'videoWidth=', els.video.videoWidth,
                  'src=', els.video.src ? els.video.src.slice(0, 50) : 'null');
      setStatus('视频还没播（可能浏览器解码不了此 mp4）。请手动点一下视频自带 ▶，或换 mp4(H.264) 重试');
    }
  }, 5000);
}

function startVideoPlayback() { doStart(); }  // legacy compatibility

function startLoop() {
  stopLoop();
  loopTimer = setInterval(tick, parseInt(els.interval.value || '100'));
}
function stopLoop() {
  if (loopTimer) { clearInterval(loopTimer); loopTimer = null; }
}

async function tick() {
  if (inflight) return;
  if (!els.video.src) {
    showVideoError('请先选视频', '还没有选视频文件');
    return;
  }
  if (els.video.paused && els.video.readyState === 0) {
    showVideoError('视频解码失败', 'readyState=0 表示浏览器没识别这个视频文件。' +
                  (els.video.error ? 'error.code=' + els.video.error.code +
                   ' (' + ({1:'ABORTED',2:'NETWORK',3:'DECODE',4:'SRC_NOT_SUPPORTED'}[els.video.error.code] || '?') + ')' : '') +
                  '——请换 mp4 (H.264 Baseline) 重试');
    return;
  }
  if (els.video.paused) {
    // 暂停中
    showVideoError('视频已暂停', '请点视频下方的 ▶ 按钮开始播放');
    return;
  }
  if (!els.video.videoWidth) {
    showVideoError('视频无尺寸', 'videoWidth=0，请换 mp4 (H.264)');
    return;
  }
  hideVideoError();
  const t0 = performance.now();
  inflight = true;
  try
  {
    const blob = await grabFrame();
    if (!blob) return;
    const fd = new FormData();
    fd.append('frame', blob, 'frame.jpg');
    fd.append('conf', els.conf.value || '0.25');
    fd.append('method', els.method.value || 'stgcn');
    const r = await fetch('/api/live_predict', { method: 'POST', body: fd });
    const data = await r.json();
    if (data.error) {
      setStatus('错误: ' + data.error);
      return;
    }
    onResult(data, t0);
  }
  catch (e) {
    console.error(e);
  }
  finally {
    inflight = false;
  }
}

let offCanvas = null, offCtx = null;   // 复用的离屏抽帧 canvas（避免每帧新建 → GC 卡顿）
async function grabFrame() {
  if (!els.video.videoWidth) return null;
  const w = els.video.videoWidth, h = els.video.videoHeight;
  if (!offCanvas) { offCanvas = document.createElement('canvas'); offCtx = offCanvas.getContext('2d'); }
  offCanvas.width = w; offCanvas.height = h;
  offCtx.drawImage(els.video, 0, 0, w, h);
  return new Promise(res => offCanvas.toBlob(b => res(b), 'image/jpeg', 0.85));
}

function onResult(data, t0) {
  frameIdx++;
  drawSkeleton(data.kpts, data.image_size, data.box, data.topk, data.provisional);

  // FPS
  const dt = performance.now() - t0;
  const instFps = 1000 / Math.max(dt, 1);
  fpsAvg = fpsAvg * 0.7 + instFps * 0.3;
  els.frameCount.textContent = frameIdx;
  els.fps.textContent = fpsAvg.toFixed(1);
  els.fpsBadge.textContent = 'FPS ' + fpsAvg.toFixed(0);

  const buf = data.buffer_len || 0;
  const need = data.buffer_need || 300;
  const phase = data.phase || 'warming';

  if (phase === 'warming') {
    setWarming(buf, need);
    els.warming.classList.remove('hidden');
    els.phaseBadge.textContent = `缓冲中 ${buf}/${need}`;
    els.resultCard.classList.add('hidden');
  } else {  // ready
    els.warming.classList.add('hidden');
    els.phaseFill.style.width = '100%';
    els.phaseFill.classList.add('ready');
    els.phaseDetail.textContent = '就绪 · 每 30 帧滑动更新';
    if (els.warmingBarFill) els.warmingBarFill.style.width = '100%';
    els.phaseBadge.textContent = '推理就绪';
    els.nextInfer.textContent = data.next_infer_in != null ? `${data.next_infer_in} 帧后` : '即将触发';
    // Top1
    if (data.top1) {
      setResult(`${data.top1.name} (label ${data.top1.label})`, data.top1.confidence);
    }
  }
  setStatus(`mode=${data.method || 'stgcn'} · ${phase}`);
}

function setWarming(buf, need) {
  const pct = Math.min(buf / need * 100, 100);
  els.phaseFill.style.width = pct.toFixed(0) + '%';
  els.phaseFill.classList.remove('ready');
  els.phaseDetail.textContent = `正在累积骨架 · ${buf} / ${need} 帧`;
  els.warmingProgress.textContent = `${buf} / ${need}`;
  // 大进度条 (视频画面内的进度条)
  if (els.warmingBarFill) {
    els.warmingBarFill.style.width = pct.toFixed(0) + '%';
  }
}

function drawSkeleton(people, imageSize, box, topk, provisional) {
  const ctx = els.canvas.getContext('2d');
  const cw = els.canvas.width, ch = els.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  // video 是 object-fit: contain，可能有黑边 → 骨架/框坐标必须按实际显示区域换算
  const rect = videoDisplayRect();
  const px = (u, v) => [rect.x + u * rect.w, rect.y + v * rect.h];

  // 1) 施救者检测框 + 框上方类别标签（最多显示 Top-2 最可能的类别）
  if (box) {
    const [x1, y1] = px(box.x1, box.y1);
    const [x2, y2] = px(box.x2, box.y2);
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // 收集 Top-2 类别（topk 可能是数组；兼容旧的单 top1 对象）
    let items = [];
    if (Array.isArray(topk)) {
      items = topk.filter(it => it && it.name && it.confidence > 0);
    } else if (topk && topk.name) {
      items = [topk];
    }
    items = items.slice(0, 2);

    ctx.font = '600 12px system-ui, sans-serif';
    const lh = 19;                  // 每行标签高度
    const pad = 7;
    if (items.length === 0) {
      // 完全没类别 → 只显示「施救者」占位
      items = [{ name: '施救者', confidence: null }];
    }

    // 逐行构建文案，量最大宽度（Top-2 两行背景等宽对齐）
    const rows = items.map((it, i) => {
      const pct = it.confidence != null ? `${(it.confidence * 100).toFixed(0)}%` : '';
      const main = it.name + (pct ? ` ${pct}` : '');
      return i === 0 && provisional ? `${main} 估` : main;
    });
    let tw = 0;
    rows.forEach(t => { tw = Math.max(tw, ctx.measureText(t).width); });
    tw += pad * 2;
    const totalH = rows.length * lh;
    // 标签整体贴框上方；放不下则贴框内顶部
    let ly = y1 - totalH - 2;
    if (ly < 0) ly = y1 + 2;

    // 半透明底 + 两行（Top1 主色 / Top2 次要色）
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(x1, ly, tw, totalH);
    rows.forEach((txt, i) => {
      const ry = ly + i * lh;
      if (i === 0) {
        ctx.fillStyle = provisional ? 'rgba(255,179,0,0.94)' : 'rgba(0,230,118,0.94)';
        ctx.fillRect(x1, ry, tw, lh);
        ctx.fillStyle = provisional ? '#1a1000' : '#001a0f';
      } else {
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillRect(x1, ry, tw, lh);
        ctx.fillStyle = '#c9d1d9';
      }
      ctx.textBaseline = 'middle';
      ctx.fillText(txt, x1 + pad, ry + lh / 2 + 0.5);
    });
    ctx.textBaseline = 'alphabetic';
  }

  // 2) 骨架
  const colors = ['#58a6ff', '#f97583'];
  people.forEach((p, mi) => {
    const color = colors[mi % colors.length];
    EDGES.forEach(([a, b]) => {
      const pa = p[a], pb = p[b];
      if (!pa || !pb || pa.score < 0.05 || pb.score < 0.05) return;
      const [ax, ay] = px(pa.x, pa.y);
      const [bx, by] = px(pb.x, pb.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    });
    p.forEach(pt => {
      if (pt.score < 0.05) return;
      const r = 3 + pt.score * 4;
      const [cx, cy] = px(pt.x, pt.y);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.35 + pt.score * 0.65;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 1.0;
      ctx.stroke();
    });
  });
  ctx.globalAlpha = 1.0;
}

/** video 在 canvas 内的实际显示区域（处理 object-fit: contain 的黑边） */
function videoDisplayRect() {
  const vw = els.video.videoWidth || 0, vh = els.video.videoHeight || 0;
  const cw = els.canvas.width, ch = els.canvas.height;
  if (!vw || !vh || !cw || !ch) return { x: 0, y: 0, w: cw, h: ch };
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale, dh = vh * scale;
  return { x: (cw - dw) / 2, y: (ch - dh) / 2, w: dw, h: dh };
}

function setResult(text, conf) {
  els.resultCard.classList.remove('hidden');
  els.result.textContent = text;
  els.confFill.style.width = Math.min((conf || 0) * 100, 100).toFixed(0) + '%';
}

function setStatus(text) { els.status.textContent = text; }