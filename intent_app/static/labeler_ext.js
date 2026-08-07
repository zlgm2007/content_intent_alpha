/* 数据标注页 - 外部数据同步脚本（labeler_ext.js） */
let syncPollTimer = null;
let syncAnnPage = 1;
let detailWorkId = null;   // 当前详情弹窗打开的外部作品ID（标注后刷新弹窗用）

// ---- 同步方式切换 ----
function onSyncModeChange() {
  const m = document.getElementById('syncMode').value;
  document.getElementById('fieldDays').style.display = m === 'days' ? '' : 'none';
  document.getElementById('fieldLimit').style.display = m === 'limit' ? '' : 'none';
}

// ---- 测试连接 ----
async function testConnection() {
  const data = await api('/sync/api/test_connection');
  const el = document.getElementById('esConnMsg');
  el.innerHTML = data.ok
    ? `<span class="stat green">${esc(data.message)}</span>`
    : `<span class="stat red">${esc(data.message)}</span>`;
}

// ---- 开始 / 停止同步 ----
async function startSync() {
  if (!currentGoalId) { alert('请先选择意图目标'); return; }
  const body = {goal_id: parseInt(currentGoalId), indices: document.getElementById('syncIndex').value};
  if (document.getElementById('syncMode').value === 'limit') {
    const limit = parseInt(document.getElementById('syncLimit').value) || 10000;
    if (limit < 1 || limit > 100000) { alert('同步条数需在 1-100000 之间'); return; }
    body.limit = limit;
  } else {
    const days = parseInt(document.getElementById('syncDays').value) || 1;
    if (days < 1 || days > 30) { alert('同步天数需在 1-30 之间'); return; }
    body.days = days;
  }
  const data = await api('/sync/api/start', { method:'POST', body: JSON.stringify(body) });
  if (data.error) { alert(data.error); return; }
  document.getElementById('btnSyncStart').disabled = true;
  document.getElementById('btnSyncStop').disabled = false;
  document.getElementById('syncProgressCard').style.display = 'block';
  startSyncPolling();
}

async function stopSync() {
  if (!confirm('确定停止同步？')) return;
  await api('/sync/api/stop', { method:'POST', body: '{}' });
}

// ---- 进度轮询 ----
function startSyncPolling() {
  if (syncPollTimer) clearInterval(syncPollTimer);
  syncPollTimer = setInterval(pollSync, 1000);
  pollSync();
}

async function pollSync() {
  const data = await api('/sync/api/status');
  const p = data.progress || {};
  const pct = p.percent || 0;
  document.getElementById('syncProgressFill').style.width = pct + '%';
  document.getElementById('syncProgressFill').textContent = pct + '%';
  const phaseTxt = p.phase === 'done' ? '完成'
    : p.phase === 'error' ? '出错'
    : p.phase === 'querying' ? '查询中'
    : p.phase === 'writing' ? '写入中' : (p.index || '准备中');
  document.getElementById('syncPhase').textContent = phaseTxt;
  document.getElementById('syncDocs').textContent = `${p.docs_processed}/${p.docs_total ?? '-'}`;
  document.getElementById('syncWorks').textContent = `${p.works_inserted||0}/${p.works_skipped||0}`;
  document.getElementById('syncAnn').textContent = `${p.ann_inserted||0}/${p.ann_skipped||0}`;
  document.getElementById('syncElapsed').textContent = fmtElapsed(p.elapsed);

  const logs = data.logs || [];
  const logBox = document.getElementById('syncLogBox');
  logBox.innerHTML = logs.map(l => {
    let cls = l.includes('ERROR') ? 'log-err'
      : (l.includes('同步结束') || l.includes('完成')) ? 'log-ok' : '';
    return `<div class="${cls}">${esc(l)}</div>`;
  }).join('');
  logBox.scrollTop = logBox.scrollHeight;

  if (data.state !== 'running') {
    clearInterval(syncPollTimer);
    syncPollTimer = null;
    document.getElementById('btnSyncStart').disabled = false;
    document.getElementById('btnSyncStop').disabled = true;
    loadSyncHistory();
    loadSyncedAnnotations();
    loadContents();   // 同步完成刷新作品列表（合并视图能看到外部作品）
  }
}

function fmtElapsed(sec) {
  if (!sec) return '-';
  if (sec < 60) return Math.round(sec) + 's';
  return Math.floor(sec/60) + 'm' + Math.round(sec%60) + 's';
}

// ---- 同步历史 ----
async function loadSyncHistory() {
  if (!currentGoalId) return;
  const data = await api(`/sync/api/history?goal_id=${currentGoalId}&size=10`);
  const el = document.getElementById('syncHistory');
  if (!data.items || data.items.length === 0) {
    el.innerHTML = '<div class="empty">暂无同步记录</div>';
    return;
  }
  el.innerHTML = '<table><thead><tr>' +
    '<th>批次</th><th>索引</th><th>范围</th><th>状态</th><th>作品(增/跳)</th><th>评论(增/跳)</th><th>时间</th>' +
    '</tr></thead><tbody>' + data.items.map(h => {
      const st = h.state;
      const stCls = st==='done'?'green':st==='error'?'red':st==='running'?'blue':'yellow';
      const stTxt = st==='done'?'完成':st==='error'?'出错':st==='running'?'进行中':st;
      const rangeTxt = h.sync_limit ? `最近 ${h.sync_limit} 条` : `最近 ${h.days} 天`;
      return `<tr>
        <td>#${h.id}</td>
        <td>${esc(h.indices)}</td>
        <td>${rangeTxt}</td>
        <td><span class="stat ${stCls}">${stTxt}</span></td>
        <td>${h.works_inserted}/${h.works_skipped}</td>
        <td>${h.ann_inserted}/${h.ann_skipped}</td>
        <td style="font-size:12px;">${esc(h.finished_at||h.started_at||'')}</td>
      </tr>`;
    }).join('') + '</tbody></table>';
}

// ---- 外部评论标注（供外部页签与作品详情弹窗复用） ----
function annotationHtml(a) {
  let scoreHtml = '';
  for (let s = 0; s <= 5; s++) {
    const cls = s >= 3 ? 's-high' : 's-low';
    const active = a.score === s ? 'active' : '';
    scoreHtml += `<button class="score-btn ${cls} ${active}" onclick="labelSynced(${a.id}, ${s})">${s}</button>`;
  }
  const rawBadge = a.raw_score != null
    ? `<span class="stat ${a.raw_score>=3?'green':'gray'}">ES ${a.raw_score}分</span>`
    : `<span class="stat gray">未打分</span>`;
  const labelBadge = a.status === 'labeled'
    ? `<span class="stat ${a.score>=3?'green':'gray'}">标注 ${a.score}分</span>`
    : a.status === 'skipped' ? `<span class="stat yellow">已跳过</span>`
    : `<span class="stat yellow">未标注</span>`;
  return `
    <div class="comment-item">
      <div class="c-text">${esc(a.comment)}</div>
      <div class="c-meta">
        ${rawBadge} ${labelBadge}
        <span style="font-size:12px; color:#8a9199;">${esc(a.source_index||'')}</span>
        <div class="actions" style="margin-left:auto;">
          ${scoreHtml}
          ${a.status==='labeled' ? `<button class="btn btn-sm" onclick="unlabelSynced(${a.id})">取消标注</button>` : ''}
          ${a.status!=='skipped' ? `<button class="btn btn-sm" onclick="skipSynced(${a.id})">跳过</button>` : ''}
        </div>
      </div>
    </div>`;
}

// ---- 已同步评论标注列表（外部数据页签） ----
async function loadSyncedAnnotations() {
  if (!currentGoalId) return;
  const status = document.getElementById('syncStatusFilter').value;
  const data = await api(`/sync/api/annotations?goal_id=${currentGoalId}&status=${status}&page=${syncAnnPage}&size=10`);
  const el = document.getElementById('syncedList');
  if (!data.items || data.items.length === 0) {
    el.innerHTML = '<div class="empty">暂无同步评论，请先执行同步</div>';
    document.getElementById('syncedPager').innerHTML = '';
    return;
  }
  el.innerHTML = data.items.map(a => {
    const workTxt = (a.note_title || a.work_content || '').substring(0, 60);
    return `
      <div>
        <div style="font-size:12px; color:#656d76; margin-bottom:4px;">
          作品: ${esc(workTxt)}... <span style="color:#8a9199;">#${esc(a.note_id||'')}</span>
        </div>
        ${annotationHtml(a)}
      </div>`;
  }).join('');
  renderPager('syncedPager', data.page, Math.ceil(data.total/data.size), (p) => {
    syncAnnPage = p; loadSyncedAnnotations();
  });
}

async function labelSynced(annId, score) {
  const data = await api('/sync/api/label', {
    method:'POST', body: JSON.stringify({id: annId, score})
  });
  if (data.error) { alert(data.error); return; }
  loadSyncedAnnotations();
  if (detailWorkId) showExternalDetail(detailWorkId);
}

async function unlabelSynced(annId) {
  const data = await api('/sync/api/unlabel', {
    method:'POST', body: JSON.stringify({id: annId})
  });
  if (data.error) { alert(data.error); return; }
  loadSyncedAnnotations();
  if (detailWorkId) showExternalDetail(detailWorkId);
}

async function skipSynced(annId) {
  const data = await api('/sync/api/skip', {
    method:'POST', body: JSON.stringify({id: annId})
  });
  if (data.error) { alert(data.error); return; }
  loadSyncedAnnotations();
  if (detailWorkId) showExternalDetail(detailWorkId);
}

// ---- 外部作品详情弹窗（作品列表合并视图进入） ----
async function showExternalDetail(workId) {
  detailWorkId = workId;
  const data = await api(`/sync/api/work_detail?id=${workId}`);
  if (data.error) { alert(data.error); return; }
  const w = data.work;
  const anns = data.annotations || [];
  let html = `
    <div class="card" style="margin-bottom:16px;">
      <h3>外部作品 #${w.id} ${esc(w.note_title||'')} <span class="stat blue">${esc(w.source_index||'')}</span></h3>
      <p style="white-space:pre-wrap; margin-top:8px; font-size:12px; color:#656d76;">${esc(w.content||'')}</p>
    </div>
    <h3 style="margin-top:16px;">评论标注 (${anns.length})</h3>
  `;
  if (anns.length === 0) {
    html += '<div class="empty">暂无评论</div>';
  } else {
    html += anns.map(annotationHtml).join('');
  }
  document.getElementById('detailContent').innerHTML = html;
  document.getElementById('detailModal').classList.remove('hidden');
}
