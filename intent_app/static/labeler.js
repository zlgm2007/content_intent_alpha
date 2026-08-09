/* 数据标注页脚本（labeler.html） */
let currentGoalId = null;
let contentPage = 1;
let unlabeledPage = 1;
let pollTimer = null;
let autoPollTimer = null;
// 分数配置（从意图目标加载）
let goalMaxScore = 5;
let goalThreshold = 3;
let goalScoreDefs = null;

async function api(path, opts={}) {
  const res = await fetch(path, { headers: {'Content-Type':'application/json'}, ...opts });
  return res.json();
}

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ---- 意图目标加载 ----
async function loadGoals() {
  const data = await api('/intent/api/list');
  const sel = document.getElementById('goalSelect');
  const oldVal = sel.value;
  sel.innerHTML = '<option value="">-- 请选择 --</option>' +
    (data.goals||[]).map(g => `<option value="${g.id}">${esc(g.name)}</option>`).join('');
  if (oldVal) sel.value = oldVal;
  if (currentGoalId) sel.value = currentGoalId;
}

function onGoalChange() {
  currentGoalId = document.getElementById('goalSelect').value;
  if (!currentGoalId) return;
  contentPage = 1;
  unlabeledPage = 1;
  syncAnnPage = 1;
  loadGoalConfig().then(() => {
    loadStats();
    loadContents();
    loadUnlabeled();
    loadSyncHistory();
    loadSyncedAnnotations();
    loadAutoLabelModels();
  });
}

// ---- 加载意图目标的分数配置 ----
async function loadGoalConfig() {
  const data = await api(`/intent/api/detail?id=${currentGoalId}`);
  if (data.error) return;
  const defs = data.score_definitions;
  if (defs && Array.isArray(defs) && defs.length >= 2) {
    goalMaxScore = Math.max(...defs.map(d => d.score));
    goalScoreDefs = defs;
  } else {
    goalMaxScore = 5;
    goalScoreDefs = null;
  }
  goalThreshold = data.intent_threshold != null ? data.intent_threshold : Math.ceil((goalMaxScore + 1) / 2);
  renderScoreFilter();
  // 更新提示文字
  const hint = document.getElementById('quickLabelHint');
  if (hint) {
    hint.textContent = `为每条评论选择 0-${goalMaxScore} 分，${goalThreshold} 分以上为有意图，点击分数即保存。`;
  }
}

// ---- 分数按钮 HTML 生成 ----
function scoreButtonsHtml(commentId, currentScore, onclickFmt) {
  let html = '';
  for (let s = 0; s <= goalMaxScore; s++) {
    const cls = s >= goalThreshold ? 's-high' : 's-low';
    const active = currentScore === s ? ' active' : '';
    const onclick = onclickFmt.replace('{id}', commentId).replace('{score}', s);
    html += `<button class="score-btn ${cls}${active}" onclick="${onclick}" title="${getScoreDesc(s)}">${getScoreLabel(s)}</button>`;
  }
  return html;
}

function getScoreLabel(score) {
  if (goalScoreDefs) {
    const d = goalScoreDefs.find(x => x.score === score);
    if (d) return `${score} ${d.label}`;
  }
  return String(score);
}

function getScoreDesc(score) {
  if (goalScoreDefs) {
    const d = goalScoreDefs.find(x => x.score === score);
    if (d && d.desc) return d.desc;
  }
  return `${score} 分`;
}

// ---- 评分筛选（动态） ----
function renderScoreFilter() {
  const box = document.getElementById('scoreFilterBox');
  if (!box) return;
  let html = `<label style="font-size:12px;"><input type="checkbox" value="none" checked onchange="onQuickFilter()"> 未标注</label>`;
  for (let s = 0; s <= goalMaxScore; s++) {
    html += `<label style="font-size:12px;"><input type="checkbox" value="${s}" onchange="onQuickFilter()"> ${getScoreLabel(s)}</label>`;
  }
  box.innerHTML = html;
}

async function loadStats() {
  if (!currentGoalId) return;
  const data = await api(`/lbl/api/stats?goal_id=${currentGoalId}`);
  const box = document.getElementById('statsBox');
  const dist = data.score_distribution || {};
  let distHtml = Object.entries(dist).map(([s,c]) =>
    `<span class="stat ${parseInt(s)>=goalThreshold?'green':'gray'}">${getScoreLabel(parseInt(s))}:${c}</span>`
  ).join(' ');
  box.innerHTML = `
    <span class="stat blue">总评论 ${data.total}</span>
    <span class="stat green">已标注 ${data.labeled}</span>
    <span class="stat yellow">未标注 ${data.unlabeled}</span>
    ${distHtml}
  `;
}

// ---- Tab 切换 ----
function switchTab(tab) {
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab));
  ['contents','autolabel','quickLabel','import','external'].forEach(t =>
    document.getElementById('tab-'+t).classList.toggle('hidden', t !== tab));
}

// ---- 作品列表 ----
async function loadContents() {
  if (!currentGoalId) return;
  const data = await api(`/lbl/api/contents?goal_id=${currentGoalId}&page=${contentPage}&size=20`);
  const el = document.getElementById('contentList');
  if (!data.contents || data.contents.length === 0) {
    el.innerHTML = '<div class="empty">暂无作品，请先添加</div>';
    document.getElementById('contentPager').innerHTML = '';
    return;
  }
  el.innerHTML = '<table><thead><tr>' +
    '<th>ID</th><th>来源</th><th>作品内容</th><th>评论</th><th>已标注</th><th>操作</th>' +
    '</tr></thead><tbody>' + data.contents.map(c => `
    <tr>
      <td>${c.id}</td>
      <td><span class="stat ${c.source==='es'?'blue':'gray'}">${c.source==='es'?'外部':'手工'}</span></td>
      <td style="max-width:400px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${esc(c.text)}">${esc(c.text)}</td>
      <td><span class="stat blue">${c.comment_count}</span></td>
      <td><span class="stat ${c.labeled_count>0?'green':'gray'}">${c.labeled_count}</span></td>
      <td><div class="actions">
        <button class="btn btn-sm btn-primary" onclick="showDetail(${c.id}, '${c.source}')">详情/标注</button>
        <button class="btn btn-sm btn-danger" onclick="deleteContent(${c.id}, '${c.source}')">删除</button>
      </div></td>
    </tr>`).join('') + '</tbody></table>';
  renderPager('contentPager', data.page, Math.ceil(data.total/data.size), (p) => {
    contentPage = p; loadContents();
  });
}

async function addContent() {
  const text = document.getElementById('contentText').value.trim();
  if (!text) { alert('请输入作品内容'); return; }
  if (!currentGoalId) { alert('请先选择意图目标'); return; }
  const data = await api('/lbl/api/content_create', {
    method:'POST', body: JSON.stringify({goal_id: parseInt(currentGoalId), text})
  });
  if (data.error) { alert(data.error); return; }
  document.getElementById('contentText').value = '';
  loadContents();
  loadStats();
}

async function deleteContent(id, source) {
  if (!confirm('删除作品将同时删除其所有评论，确定？')) return;
  const path = source === 'es' ? '/sync/api/work_delete' : '/lbl/api/content_delete';
  const data = await api(path, { method:'POST', body: JSON.stringify({id}) });
  if (data.error) { alert(data.error); return; }
  loadContents();
  loadStats();
}

// ---- 作品详情 + 评论标注 ----
async function showDetail(contentId, source) {
  if (source === 'es') { showExternalDetail(contentId); return; }
  const data = await api(`/lbl/api/content_detail?id=${contentId}`);
  if (data.error) { alert(data.error); return; }
  const c = data.content;
  const comments = data.comments || [];
  let html = `
    <div class="card" style="margin-bottom:16px;">
      <h3>作品 #${c.id}</h3>
      <p style="white-space:pre-wrap; margin-top:8px;">${esc(c.text)}</p>
    </div>
    <div class="row">
      <div class="field" style="flex:3;">
        <label>添加评论（每行一条）</label>
        <textarea id="batchComments" placeholder="评论1&#10;评论2&#10;评论3"></textarea>
      </div>
      <button class="btn btn-primary" onclick="addBatchComments(${c.id})">批量添加</button>
    </div>
    <h3 style="margin-top:16px;">评论列表 (${comments.length})</h3>
  `;
  if (comments.length === 0) {
    html += '<div class="empty">暂无评论</div>';
  } else {
    html += comments.map(cm => {
      const scoreHtml = scoreButtonsHtml(cm.id, cm.score, `labelComment(${cm.id}, {score}, ${c.id})`);
      const labelBadge = cm.status === 'labeled'
        ? `<span class="stat ${cm.score>=goalThreshold?'green':'gray'}">${cm.score}分 / ${cm.score>=goalThreshold?'有意图':'无意图'}</span>`
        : `<span class="stat yellow">未标注</span>`;
      return `
        <div class="comment-item">
          <div class="c-text">${esc(cm.comment)}</div>
          <div class="c-meta">
            ${labelBadge}
            <span>#${cm.id}</span>
            <div class="actions" style="margin-left:auto;">
              ${scoreHtml}
              ${cm.status==='labeled' ? `<button class="btn btn-sm" onclick="unlabelComment(${cm.id}, ${c.id})">取消标注</button>` : ''}
              <button class="btn btn-sm btn-danger" onclick="deleteComment(${cm.id}, ${c.id})">删除</button>
            </div>
          </div>
        </div>`;
    }).join('');
  }
  document.getElementById('detailContent').innerHTML = html;
  document.getElementById('detailModal').classList.remove('hidden');
}

function closeDetail() {
  document.getElementById('detailModal').classList.add('hidden');
  detailWorkId = null;
}

async function addBatchComments(contentId) {
  const text = document.getElementById('batchComments').value.trim();
  if (!text) { alert('请输入评论'); return; }
  const comments = text.split('\n').map(s => s.trim()).filter(s => s);
  const data = await api('/lbl/api/comment_batch', {
    method:'POST', body: JSON.stringify({
      goal_id: parseInt(currentGoalId),
      content_id: contentId,
      comments
    })
  });
  if (data.error) { alert(data.error); return; }
  showDetail(contentId);
  loadContents();
  loadStats();
}

async function labelComment(commentId, score, contentId) {
  const data = await api('/lbl/api/comment_label', {
    method:'POST', body: JSON.stringify({id: commentId, score})
  });
  if (data.error) { alert(data.error); return; }
  showDetail(contentId);
  loadStats();
}

async function unlabelComment(commentId, contentId) {
  const data = await api('/lbl/api/comment_unlabel', {
    method:'POST', body: JSON.stringify({id: commentId})
  });
  if (data.error) { alert(data.error); return; }
  showDetail(contentId);
  loadStats();
}

async function deleteComment(commentId, contentId) {
  if (!confirm('确定删除这条评论？')) return;
  const data = await api('/lbl/api/comment_delete', {
    method:'POST', body: JSON.stringify({id: commentId})
  });
  if (data.error) { alert(data.error); return; }
  showDetail(contentId);
  loadContents();
  loadStats();
}

// ---- 快速标注 ----
function quickFilterScores() {
  const sel = [...document.querySelectorAll('#tab-quickLabel input[type=checkbox]:checked')]
    .map(c => c.value);
  // 全不选时用显式标记，避免空参数被 URL 解析丢弃（后端会按无效组合返回空结果）
  return sel.length ? sel.join(',') : '__empty';
}

function onQuickFilter() {
  unlabeledPage = 1;
  loadUnlabeled();
}

async function loadUnlabeled() {
  if (!currentGoalId) return;
  const data = await api(`/lbl/api/unlabeled?goal_id=${currentGoalId}&page=${unlabeledPage}&size=20&scores=${quickFilterScores()}`);
  const el = document.getElementById('unlabeledList');
  if (!data.items || data.items.length === 0) {
    el.innerHTML = '<div class="empty">没有符合条件的评论</div>';
    document.getElementById('unlabeledPager').innerHTML = '';
    return;
  }
  el.innerHTML = data.items.map(cm => {
    const scoreHtml = scoreButtonsHtml(cm.id, cm.score, `quickLabel('${cm.source}', ${cm.id}, {score})`);
    const srcBadge = cm.source === 'es'
      ? `<span class="stat blue">外部</span>`
      : `<span class="stat gray">手工</span>`;
    const rawBadge = cm.source === 'es' && cm.raw_score != null
      ? `<span class="stat ${cm.raw_score>=goalThreshold?'green':'gray'}">ES ${cm.raw_score}分</span>`
      : '';
    return `
      <div class="comment-item">
        <div style="font-size:12px; color:#656d76; margin-bottom:4px;">作品: ${esc(cm.content_text||'').substring(0,80)}...</div>
        <div class="c-text">${esc(cm.comment)}</div>
        <div class="c-meta">
          ${srcBadge} ${rawBadge}
          <span>#${cm.id}</span>
          <div class="actions" style="margin-left:auto;">
            ${scoreHtml}
            <button class="btn btn-sm" onclick="quickSkip('${cm.source}', ${cm.id})">跳过</button>
          </div>
        </div>
      </div>`;
  }).join('');
  renderPager('unlabeledPager', data.page, Math.ceil(data.total/data.size), (p) => {
    unlabeledPage = p; loadUnlabeled();
  });
}

async function quickLabel(source, commentId, score) {
  const path = source === 'es' ? '/sync/api/label' : '/lbl/api/comment_label';
  const data = await api(path, {method:'POST', body: JSON.stringify({id: commentId, score})});
  if (data.error) { alert(data.error); return; }
  loadUnlabeled();
  loadStats();
}

async function quickSkip(source, commentId) {
  if (source === 'es') {
    const data = await api('/sync/api/skip', {
      method:'POST', body: JSON.stringify({id: commentId})
    });
    if (data.error) { alert(data.error); return; }
  }
  loadUnlabeled();
}

// ---- 自动标注 ----
async function loadAutoLabelModels() {
  const sel = document.getElementById('autoModelSelect');
  if (!currentGoalId) {
    sel.innerHTML = '<option value="">-- 请选择 --</option>';
    return;
  }
  const data = await api(`/train/api/models?goal_id=${currentGoalId}`);
  const models = (data.models||[]).filter(m => m.status === 'trained' && m.onnx_exists);
  sel.innerHTML = '<option value="">-- 请选择 --</option>' +
    models.map(m => `<option value="${m.id}">#${m.id} ${esc(m.name||'')}</option>`).join('');
  // 默认选中最近训练的模型（API 按 id 倒序）
  if (models.length > 0) sel.value = models[0].id;
}

async function startAutoLabel() {
  if (!currentGoalId) { alert('请先选择意图目标'); return; }
  const modelId = document.getElementById('autoModelSelect').value;
  if (!modelId) { alert('请选择已训练模型'); return; }
  const threshold = parseFloat(document.getElementById('autoThreshold').value);
  const maxCount = parseInt(document.getElementById('autoMaxCount').value) || 0;
  if (!(threshold >= 0.5 && threshold <= 1.0)) { alert('阈值必须在 0.5-1.0 之间'); return; }
  const data = await api('/autolabel/api/start', {
    method:'POST', body: JSON.stringify({
      goal_id: parseInt(currentGoalId), model_id: parseInt(modelId),
      threshold, max_count: maxCount
    })
  });
  const msg = document.getElementById('autoMsg');
  if (data.error) { msg.innerHTML = `<span class="stat red">${esc(data.error)}</span>`; return; }
  msg.innerHTML = `<span class="stat green">已启动</span>`;
  document.getElementById('autoProgressCard').style.display = 'block';
  document.getElementById('autoLogBox').innerHTML = '';
  document.getElementById('btnAutoStart').disabled = true;
  document.getElementById('btnAutoStop').disabled = false;
  pollAutoLabel();
}

async function stopAutoLabel() {
  await api('/autolabel/api/stop', { method:'POST', body: '{}' });
}

function pollAutoLabel() {
  clearTimeout(autoPollTimer);
  autoPollTimer = setTimeout(async () => {
    const data = await api('/autolabel/api/status');
    const p = data.progress || {};
    document.getElementById('autoProcessed').textContent = `${p.processed||0}/${p.total||0}`;
    document.getElementById('autoLabeled').textContent = p.labeled||0;
    document.getElementById('autoLowConf').textContent = p.low_conf||0;
    document.getElementById('autoElapsed').textContent = fmtElapsed(p.elapsed||0);
    const fill = document.getElementById('autoProgressFill');
    fill.style.width = (p.percent||0) + '%';
    fill.textContent = (p.percent||0) + '%';
    const logBox = document.getElementById('autoLogBox');
    logBox.innerHTML = (data.logs||[]).map(l => `<div>${esc(l)}</div>`).join('');
    logBox.scrollTop = logBox.scrollHeight;
    if (data.state === 'running' || data.state === 'idle') {
      pollAutoLabel();
    } else {
      document.getElementById('btnAutoStart').disabled = false;
      document.getElementById('btnAutoStop').disabled = true;
      document.getElementById('autoMsg').innerHTML =
        `<span class="stat ${data.state==='done'?'green':'red'}">${esc(data.reason||data.state)}</span>`;
      loadStats();
      loadUnlabeled();
    }
  }, 1000);
}

// ---- 批量导入 ----
async function importData() {
  const raw = document.getElementById('importJson').value.trim();
  if (!raw) { alert('请粘贴 JSON 数据'); return; }
  let body;
  try { body = JSON.parse(raw); }
  catch(e) { alert('JSON 格式错误: ' + e.message); return; }
  if (!body.goal_id) body.goal_id = parseInt(currentGoalId);
  if (!body.goal_id) { alert('缺少 goal_id'); return; }
  const data = await api('/lbl/api/import_data', {
    method:'POST', body: JSON.stringify(body)
  });
  const el = document.getElementById('importResult');
  if (data.error) {
    el.innerHTML = `<span class="stat red">${esc(data.error)}</span>`;
    return;
  }
  el.innerHTML = `<span class="stat green">导入成功: ${data.contents} 个作品, ${data.comments} 条评论</span>`;
  document.getElementById('importJson').value = '';
  loadStats();
  loadContents();
}

// ---- 分页 ----
function renderPager(elId, page, totalPages, cb) {
  const el = document.getElementById(elId);
  if (totalPages <= 1) { el.innerHTML = ''; return; }
  let html = '';
  html += `<button class="btn btn-sm" ${page<=1?'disabled':''} onclick="__pager('${elId}',${page-1})">上一页</button>`;
  html += `<span> ${page} / ${totalPages} </span>`;
  html += `<button class="btn btn-sm" ${page>=totalPages?'disabled':''} onclick="__pager('${elId}',${page+1})">下一页</button>`;
  el.innerHTML = html;
  el._cb = cb;
}

function __pager(elId, page) {
  const el = document.getElementById(elId);
  if (el._cb) el._cb(page);
}

// ---- 初始化 ----
loadGoals();
