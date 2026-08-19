/* ============================================================
   YOLO + ST-GCN WebUI - 前端逻辑
   依赖：fetch / DOM 原生 API，无第三方库
   ============================================================ */

const $ = (id) => document.getElementById(id);

const els = {
  form: $('infer-form'),
  sourcePath: $('source_path'),
  file: $('file'),
  dropzone: $('dropzone'),
  tabs: document.querySelectorAll('.tab'),
  tabPanels: { path: $('tab-path'), upload: $('tab-upload') },
  submitBtn: $('submit-btn'),
  steps: $('steps'),
  placeholder: $('placeholder'),
  videoCard: $('video-card'),
  videoMeta: $('video-meta'),
  originalImg: $('original-img'),
  originalVid: $('original-vid'),
  annotatedVid: $('annotated-video'),
  videoOverlay: $('video-overlay'),
  probsCard: $('probs-card'),
  top1Badge: $('top1-badge'),
  methodCompare: $('method-compare'),
  cmpStgcnName: $('cmp-stgcn-name'),
  cmpStgcnConf: $('cmp-stgcn-conf'),
  cmpKnnName: $('cmp-knn-name'),
  cmpKnnConf: $('cmp-knn-conf'),
  probsChart: $('probs-chart'),
  skelCard: $('skel-card'),
  frameSeg: $('frame-seg'),
  skelGrid: $('skel-grid'),
  logCard: $('log-card'),
  log: $('log'),
  statusTag: $('status-tag'),
  historyList: $('history-list'),
  errorCard: $('error-card'),
  errorMsg: $('error-msg'),
};

let pollTimer = null;
let lastLogLine = 0;
let skeletonMeta = null;

/* ============== Init ============== */
init();

function init() {
  bindTabs();
  bindDropzone();
  bindSubmit();
  loadHistory();
  // 默认隐藏结果区
  showOnly(['placeholder']);
  // 加载骨架元信息（edges/joints）
  fetch('/api/skeleton_meta').then(r => r.json()).then(d => { skeletonMeta = d; });
}

/* ============== Tabs ============== */
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

/* ============== Dropzone ============== */
function bindDropzone() {
  const dz = els.dropzone;
  dz.addEventListener('click', () => els.file.click());
  ['dragenter', 'dragover'].forEach(ev => {
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); });
  });
  ['dragleave', 'drop'].forEach(ev => {
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); });
  });
  dz.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length) {
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      els.file.files = dt.files;
      onFileChosen(files[0]);
    }
  });
  els.file.addEventListener('change', e => {
    if (e.target.files[0]) onFileChosen(e.target.files[0]);
  });
}
function onFileChosen(file) {
  els.dropzone.classList.add('has-file');
  els.dropzone.querySelector('.dropzone-title').textContent = `已选择：${file.name}`;
}

/* ============== Submit / Poll ============== */
function bindSubmit() {
  els.submitBtn.addEventListener('click', submit);
}

async function submit() {
  resetUI();
  // 不依赖 <form> 标签：从所有 [name] input 手动收集
  const fd = new FormData();
  document.querySelectorAll('aside [name]').forEach(el => {
    if (el.type === 'file') {
      if (el.files && el.files.length) fd.append(el.name, el.files[0]);
    } else {
      fd.append(el.name, el.value);
    }
  });
  setSubmitting(true);

  try {
    const r = await fetch('/api/run', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || '提交失败');
    pollLoop(data.task_id);
  } catch (err) {
    showError(err.message);
    setSubmitting(false);
  }
}

function pollLoop(tid) {
  if (pollTimer) clearInterval(pollTimer);
  lastLogLine = 0;
  showOnly(['log-card']);
  setStep(1);

  pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/poll/' + tid);
      const data = await r.json();
      if (data.error === 'task not found') {
        clearInterval(pollTimer);
        showError('任务不存在或已过期');
        return;
      }
      onPoll(tid, data);
    } catch (e) { console.error('poll err', e); }
  }, 700);
}

function onPoll(tid, data) {
  // 阶段推进
  const status = data.status;
  if (status === 'pending' || status === 'running') {
    const last = (data.log || []).join('\n');
    if (last.includes('[1/4]')) setStep(2);
    if (last.includes('[2/4]')) setStep(3);
    if (last.includes('[3/4]')) setStep(4);
    if (last.includes('[4/4]')) setStep(4);
  }

  appendLog(data.log);
  els.statusTag.textContent = status;
  els.statusTag.className = 'status ' + status;

  if (status === 'done') {
    clearInterval(pollTimer);
    setStep(4, true);
    showResult(tid, data.result);
    setSubmitting(false);
    loadHistory();
  } else if (status === 'error') {
    clearInterval(pollTimer);
    showError(data.error || '任务失败');
    setSubmitting(false);
  }
}

function setStep(n, finished = false) {
  document.querySelectorAll('.step').forEach(s => {
    const k = Number(s.dataset.step);
    s.classList.remove('active', 'done');
    if (k < n) s.classList.add('done');
    else if (k === n) {
      if (finished) s.classList.add('done');
      else s.classList.add('active');
    }
  });
}

function setSubmitting(running) {
  els.submitBtn.disabled = running;
  els.submitBtn.querySelector('.btn-text').innerHTML =
    running ? '<span class="spinner"></span>运行中...' : '▶ 提交检测';
}

function appendLog(lines) {
  const sliced = lines.slice(lastLogLine);
  lastLogLine = lines.length;
  if (sliced.length === 0) return;
  els.log.textContent += sliced.join('\n') + '\n';
  els.log.scrollTop = els.log.scrollHeight;
}

function showOnly(ids) {
  ['placeholder', 'video-card', 'probs-card', 'skel-card', 'log-card', 'error-card']
    .forEach(id => {
      const e = document.getElementById(id);
      if (e) e.classList.toggle('hidden', !ids.includes(id));
    });
}

function resetUI() {
  els.log.textContent = '';
  els.probsChart.innerHTML = '';
  els.skelGrid.innerHTML = '';
  els.frameSeg.innerHTML = '';
  els.videoOverlay.textContent = '';
  els.originalImg.removeAttribute('src');
  els.originalVid.removeAttribute('src');
  els.annotatedVid.removeAttribute('src');
  els.originalImg.classList.add('hidden');
  els.originalVid.classList.add('hidden');
  els.annotatedVid.classList.remove('hidden');
  els.methodCompare.classList.add('hidden');
  els.errorCard.classList.add('hidden');
  setStep(0);
}

/* ============== Result render ============== */
async function showResult(tid, result) {
  showOnly(['video-card', 'probs-card', 'skel-card', 'log-card']);

  // 视频对比
  if (result.annotated_url) {
    els.annotatedVid.src = result.annotated_url;
    els.videoMeta.textContent = `${result.frames} 帧 · ${result.valid_detections} 有效检测 · ${result.image_size[0]}×${result.image_size[1]}`;
    const top1 = result.topk[0];
    els.videoOverlay.textContent = `Top1: ${top1.name} · ${(top1.confidence * 100).toFixed(1)}%`;
  }

  // 原始输入：如果是视频且就是用户上传的，可以直接复用 uploads/<filename>
  const source = result.source;
  if (source.includes('uploads/')) {
    const fname = source.split(/[\\/]/).pop();
    els.originalVid.src = '/uploads/' + encodeURIComponent(fname);
    els.originalVid.classList.remove('hidden');
  } else {
    // 服务器路径（图片序列或服务器上的视频）：图片序列只能抽样显示骨架，没办法原图比对
    els.originalImg.classList.remove('hidden');
    els.originalImg.alt = '服务器路径：' + source + '\n（图片序列仅展示骨架叠加在最右骨架可视化卡片中）';
  }

  // 14 类概率图
  renderProbs(result.probs, result.topk);

  // 骨架可视化
  if (result.skel_url) {
    const skel = await fetch(result.skel_url).then(r => r.json());
    renderSkeleton(skel);
  }

  // 顶部 Top1 徽章
  const top1 = result.topk[0];
  const methodBadge = result.method_label || result.method || 'ST-GCN';
  els.top1Badge.textContent = `${methodBadge}  →  ${top1.name}  ${(top1.confidence * 100).toFixed(1)}%`;
  els.top1Badge.style.background = 'rgba(249,117,131,0.18)';
  els.top1Badge.style.color = '#f97583';

  // 双路对比条：始终展示 ST-GCN + KNN 各自独立 Top1（任何 method 都显示）
  if (result.stgcn_top1 && result.knn_top1) {
    els.methodCompare.classList.remove('hidden');
    els.cmpStgcnName.textContent = result.stgcn_top1.name;
    els.cmpStgcnConf.textContent = `置信度 ${(result.stgcn_top1.confidence * 100).toFixed(1)}%`;
    els.cmpKnnName.textContent = result.knn_top1.name;
    els.cmpKnnConf.textContent = `置信度 ${(result.knn_top1.confidence * 100).toFixed(1)}%`;
  }
}

function renderProbs(probs, topk) {
  // 从小到大排序，TopK 高亮
  const ranked = [...probs].sort((a, b) => b.prob - a.prob);
  const top1 = topk[0].name, top2 = topk[1]?.name, top3 = topk[2]?.name;
  els.probsChart.innerHTML = ranked.map(p => {
    let cls = '';
    if (p.name === top1) cls = 'top1 top1-row';
    else if (p.name === top2) cls = 'top2';
    else if (p.name === top3) cls = 'top3';
    const pct = (p.prob * 100).toFixed(1);
    return `
      <div class="prob-bar-row ${cls}">
        <div class="prob-bar-name">${escapeHtml(p.name)}</div>
        <div class="prob-bar-track"><div class="prob-bar-fill ${p.name === top1 ? 'top1' : p.name === top2 ? 'top2' : p.name === top3 ? 'top3' : ''}" style="width: ${pct}%"></div></div>
        <div class="prob-bar-val">${pct}%</div>
      </div>`;
  }).join('');
}

async function renderSkeleton(skel) {
  const meta = await fetch('/api/skeleton_meta').then(r => r.json());
  skeletonMeta = meta;

  // 框选段（frame 选择器）
  els.frameSeg.innerHTML = skel.frames.map((f, i) =>
    `<div class="seg-item ${i === 0 ? 'active' : ''}" data-i="${i}">帧 ${i + 1}/${skel.frames.length} (t=${f.t})</div>`
  ).join('');
  els.frameSeg.querySelectorAll('.seg-item').forEach(b => {
    b.addEventListener('click', () => {
      els.frameSeg.querySelectorAll('.seg-item').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      renderFrame(+b.dataset.i);
    });
  });

  // 默认显示第 3 帧（中间帧），如果有的话
  const middleIdx = Math.floor(skel.frames.length / 2);
  renderAllFrames(skel);
}

function renderAllFrames(skel) {
  els.skelGrid.innerHTML = skel.frames.map((f, i) =>
    `<div class="skel-cell">
       <div class="skel-cell-title"><span>帧 t=${f.t}</span><span>${f.people.length} 人</span></div>
       ${renderSkeletonSVG(f)}
     </div>`
  ).join('');
}

function renderFrame(i) {
  // 当前默认一直展示所有帧；如果想单帧放大可改这里
  const all = document.querySelectorAll('.skel-cell');
  all.forEach((el, idx) => {
    el.style.outline = idx === i ? '2px solid var(--primary)' : 'none';
    el.style.outlineOffset = idx === i ? '-2px' : '';
  });
}

function renderSkeletonSVG(frame) {
  const W = 1, H = 1;   // viewBox 0 0 1 1（坐标已 0~1）
  const people = frame.people;
  const edges = skeletonMeta?.edges || [];
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  svg += `<rect width="100%" height="100%" fill="#0b0f14"/>`;
  people.forEach((person, mIdx) => {
    const color = mIdx === 0 ? '#58a6ff' : '#f97583';
    // edges
    edges.forEach(([a, b]) => {
      const pa = person[a], pb = person[b];
      if (!pa || !pb || pa.score < 0.05 || pb.score < 0.05) return;
      svg += `<line x1="${pa.x}" y1="${pa.y}" x2="${pb.x}" y2="${pb.y}"
                    stroke="${color}" stroke-width="0.012" stroke-opacity="0.75"/>`;
    });
    // points
    person.forEach((p) => {
      if (p.score < 0.05) return;
      const r = 0.018 + p.score * 0.025;
      svg += `<circle cx="${p.x}" cy="${p.y}" r="${r}"
                      fill="${color}" fill-opacity="${0.3 + p.score * 0.7}"
                      stroke="${color}" stroke-width="0.005"/>`;
    });
    // neck label
    if (person[1] && person[1].score >= 0.05) {
      svg += `<text x="${person[1].x}" y="${person[1].y - 0.04}"
                    fill="#c9d1d9" font-size="0.045"
                    text-anchor="middle">P${mIdx + 1}</text>`;
    }
  });
  svg += `</svg>`;
  return svg;
}

/* ============== History ============== */
async function loadHistory() {
  try {
    const r = await fetch('/api/history');
    const data = await r.json();
    if (!data.tasks.length) {
      els.historyList.innerHTML = '<li class="history-empty">暂无历史任务</li>';
      return;
    }
    els.historyList.innerHTML = data.tasks.map(t => `
      <li class="task-item" data-id="${t.task_id}">
        <div>
          <div class="task-name">${escapeHtml(t.top1)}</div>
          <div class="task-meta">${escapeHtml((t.source || '').split(/[\\/]/).pop() || '')} · ${new Date(t.started_at * 1000).toLocaleTimeString()}</div>
        </div>
        <div class="task-conf">${(t.top1_conf * 100).toFixed(1)}%</div>
      </li>
    `).join('');
    els.historyList.querySelectorAll('.task-item').forEach(li => {
      li.addEventListener('click', async () => {
        // 直接拉历史 task 的输出重新展示
        const id = li.dataset.id;
        const r = await fetch(`/api/poll/${id}`);
        const data = await r.json();
        if (data.error === 'task not found') {
          // 历史超过 1 小时清掉了，只展示标注视频
          showError('历史任务已过期（>1 小时），仅可在重新推理时查看');
          return;
        }
        showResult(id, data.result);
      });
    });
  } catch (e) { /* ignore */ }
}

/* ============== Error ============== */
function showError(msg) {
  els.errorCard.classList.remove('hidden');
  els.errorMsg.textContent = msg;
}

/* ============== Util ============== */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
