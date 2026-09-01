<template>
  <div>
    <h2 style="margin-top: 0">训练中心</h2>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header><span>NER 实体识别模型</span></template>
          <el-form label-width="90px">
            <el-form-item label="轮数 epochs"><el-input-number v-model="nerForm.epochs" :min="1" :max="20" /></el-form-item>
            <el-form-item label="批大小"><el-input-number v-model="nerForm.batch_size" :min="4" :max="128" :step="8" /></el-form-item>
            <el-form-item label="学习率"><el-input v-model="nerForm.learning_rate" style="width: 140px" /></el-form-item>
          </el-form>
          <el-button type="primary" :loading="nerLoading" @click="startNer">开始训练 NER</el-button>
          <div style="color: #909399; font-size: 12px; margin-top: 8px">需先在「数据准备」生成 NER 数据</div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="never">
          <template #header><span>意图打分模型（6 分类）</span></template>
          <el-form label-width="90px">
            <el-form-item label="轮数 epochs"><el-input-number v-model="intentForm.epochs" :min="1" :max="20" /></el-form-item>
            <el-form-item label="批大小"><el-input-number v-model="intentForm.batch_size" :min="4" :max="128" :step="8" /></el-form-item>
            <el-form-item label="学习率"><el-input v-model="intentForm.learning_rate" style="width: 140px" /></el-form-item>
          </el-form>
          <el-button type="primary" :loading="intentLoading" @click="startIntent">开始训练意图</el-button>
          <div style="color: #909399; font-size: 12px; margin-top: 8px">需先在「数据准备」生成意图数据</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" style="margin-top: 16px">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>实时训练监控</span>
          <span style="color: #909399; font-size: 12px">同一时间只允许一个训练任务</span>
        </div>
      </template>

      <div v-if="monitors.length">
        <div v-for="m in monitors" :key="m.taskId" class="monitor-card">
          <div class="monitor-head">
            <span class="monitor-title">任务 #{{ m.taskId }} · {{ m.taskType === 'ner' ? 'NER 实体识别' : '意图打分' }}</span>
            <div style="display: flex; align-items: center; gap: 12px">
              <span v-if="m.etaText" style="color: #909399; font-size: 12px">{{ m.etaText }}</span>
              <el-tag :type="statusType(m.status)" size="small">{{ statusText(m.status) }}</el-tag>
              <el-button v-if="m.status === 'running'" type="danger" size="small" @click="stopTask(m.taskId)">停止</el-button>
            </div>
          </div>
          <el-progress :percentage="Math.round(m.progress * 100)" :status="m.status === 'failed' ? 'exception' : undefined" style="margin: 8px 0" />
          <div class="log-console" :ref="(el) => setLogRef(m.taskId, el)">
            <div v-for="(l, i) in m.logs" :key="i" class="log-line">{{ l }}</div>
            <div v-if="!m.logs.length" class="log-line log-empty">等待训练开始…</div>
          </div>
        </div>
      </div>
      <el-empty v-else description="点击上方「开始训练」后，这里会实时显示进度和日志" :image-size="60" />
    </el-card>

    <el-card shadow="never" style="margin-top: 16px">
      <template #header><span>训练任务</span></template>
      <el-table :data="tasks" size="small">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="task_type" label="类型" width="90">
          <template #default="{ row }">{{ row.task_type === 'ner' ? 'NER' : '意图' }}</template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="90">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" width="180">
          <template #default="{ row }">
            <el-progress :percentage="Math.round((row.progress || 0) * 100)" :status="row.status === 'failed' ? 'exception' : undefined" />
          </template>
        </el-table-column>
        <el-table-column prop="message" label="信息" />
        <el-table-column prop="created_at" label="创建时间" width="160" />
        <el-table-column label="操作" width="90" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'running'" type="danger" size="small" @click="stopTask(row.id)">停止</el-button>
            <span v-else style="color: #c0c4cc; font-size: 12px">—</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, nextTick, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const nerForm = reactive({ epochs: 3, batch_size: 32, learning_rate: 2e-5 })
const intentForm = reactive({ epochs: 3, batch_size: 32, learning_rate: 2e-5 })
const nerLoading = ref(false)
const intentLoading = ref(false)
const tasks = ref([])

const monitors = ref([])
const esMap = new Map()
const startTimeMap = new Map()
const logRefs = {}
let timer = null

function statusType(s) {
  return { pending: 'info', running: 'warning', done: 'success', failed: 'danger', stopped: 'info', interrupted: 'info' }[s] || 'info'
}
function statusText(s) {
  return { pending: '待处理', running: '训练中', done: '完成', failed: '失败', stopped: '已停止', interrupted: '已中断' }[s] || s
}

function fmtEta(sec) {
  sec = Math.max(0, Math.round(sec))
  if (sec < 60) return `预计剩余 ${sec} 秒`
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `预计剩余 ${m} 分 ${s} 秒`
}

function setLogRef(taskId, el) {
  if (el) logRefs[taskId] = el
}

function scrollLog(taskId) {
  nextTick(() => {
    const el = logRefs[taskId]
    if (el) el.scrollTop = el.scrollHeight
  })
}

function attachMonitor(taskId, taskType) {
  if (esMap.has(taskId)) return

  const monitor = reactive({ taskId, taskType, progress: 0, status: 'running', etaText: '', logs: [] })
  monitors.value.push(monitor)
  startTimeMap.set(taskId, Date.now())

  const es = new EventSource(`/api/train/tasks/${taskId}/stream`)
  esMap.set(taskId, es)

  es.onmessage = (e) => {
    let ev
    try { ev = JSON.parse(e.data) } catch { return }

    if (ev.type === 'log' || ev.type === 'progress') {
      monitor.logs.push(ev.message || '')
      if (monitor.logs.length > 500) monitor.logs.splice(0, monitor.logs.length - 500)
      scrollLog(taskId)
    }
    if (ev.type === 'progress' && typeof ev.progress === 'number') {
      monitor.progress = ev.progress
      const st = startTimeMap.get(taskId)
      if (st && ev.progress > 0.01) {
        const elapsed = (Date.now() - st) / 1000
        monitor.etaText = fmtEta(elapsed / ev.progress * (1 - ev.progress))
      }
    }
    if (ev.type === 'done' || ev.type === 'failed' || ev.type === 'stopped') {
      monitor.logs.push(ev.message || (ev.type === 'done' ? '训练完成' : ev.type === 'stopped' ? '已停止' : '训练失败'))
      monitor.status = ev.status || (ev.type === 'done' ? 'done' : ev.type === 'stopped' ? 'stopped' : 'failed')
      if (ev.type === 'done') monitor.progress = 1
      monitor.etaText = ''
      scrollLog(taskId)
      es.close()
      esMap.delete(taskId)
      startTimeMap.delete(taskId)
      loadTasks()
      ElMessage[ev.type === 'done' ? 'success' : ev.type === 'stopped' ? 'info' : 'error'](
        ev.message || (ev.type === 'done' ? '训练完成' : ev.type === 'stopped' ? '已停止' : '训练失败')
      )
    }
  }
  es.onerror = () => {}
}

async function startNer() {
  nerLoading.value = true
  try {
    const r = await api.post('/train/ner', nerForm)
    // 后端已有任务在跑时会返回 ok=false，此时直接提示，不再挂监控
    if (r.ok === false) {
      ElMessage.warning(r.message || '无法启动训练')
      loadTasks()
      return
    }
    attachMonitor(r.task_id, 'ner')
    loadTasks()
  } finally {
    nerLoading.value = false
  }
}

async function startIntent() {
  intentLoading.value = true
  try {
    const r = await api.post('/train/intent', intentForm)
    if (r.ok === false) {
      ElMessage.warning(r.message || '无法启动训练')
      loadTasks()
      return
    }
    attachMonitor(r.task_id, 'intent')
    loadTasks()
  } finally {
    intentLoading.value = false
  }
}

async function stopTask(taskId) {
  try {
    await api.post(`/train/tasks/${taskId}/stop`)
    ElMessage.info('已发送停止请求，正在优雅停止…')
    setTimeout(loadTasks, 1000)
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function loadTasks() {
  tasks.value = await api.get('/train/tasks')
}

async function restoreRunning() {
  // 切页回来时，恢复正在运行任务的监控
  await loadTasks()
  const running = tasks.value.filter((t) => t.status === 'running')
  running.forEach((t) => attachMonitor(t.id, t.task_type))
}

onMounted(() => {
  restoreRunning()
  timer = setInterval(loadTasks, 5000)
})
onUnmounted(() => {
  timer && clearInterval(timer)
  esMap.forEach((es) => es.close())
  esMap.clear()
})
</script>

<style scoped>
.monitor-card {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 14px;
}
.monitor-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.monitor-title {
  font-size: 14px;
  font-weight: 500;
}
.log-console {
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: 'SF Mono', 'Consolas', 'Menlo', monospace;
  font-size: 12px;
  line-height: 1.7;
  border-radius: 8px;
  padding: 12px;
  height: 220px;
  overflow-y: auto;
  margin-top: 8px;
}
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
}
.log-empty {
  color: #6a9955;
}
</style>
