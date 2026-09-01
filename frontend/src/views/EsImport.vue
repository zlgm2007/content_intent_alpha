<template>
  <div>
    <h2 style="margin-top: 0">ES 数据导入</h2>

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>ES 连接配置</span>
          <div>
            <el-button size="small" @click="loadConfig">读取配置</el-button>
            <el-button size="small" type="primary" @click="saveConfig">保存配置</el-button>
            <el-button size="small" type="success" :loading="testing" @click="testConn">测试连接</el-button>
          </div>
        </div>
      </template>
      <el-form :model="config" label-width="100px">
        <el-form-item label="地址">
          <el-input v-model="hostInput" placeholder="http://10.104.214.120:9200" />
        </el-form-item>
        <el-form-item label="索引">
          <el-input v-model="indicesInput" placeholder="逗号分隔，如 lead_clue_score_ge3,lead_clue_score_lt3" />
        </el-form-item>
        <el-form-item label="每批数量">
          <el-input-number v-model="config.scroll_size" :min="100" :max="10000" :step="100" />
        </el-form-item>
        <el-form-item label="并发分片">
          <el-input-number v-model="config.slices" :min="1" :max="16" />
        </el-form-item>
        <el-form-item label="仅核心样本">
          <el-switch v-model="config.only_core" />
          <span style="margin-left: 8px; color: #909399; font-size: 12px">只拉有意图分数或实体的文档</span>
        </el-form-item>
      </el-form>
      <el-alert v-if="testResult" :type="testResult.ok ? 'success' : 'error'" :closable="false" style="margin-top: 8px">
        <template v-if="testResult.ok">
          集群 {{ testResult.cluster }} v{{ testResult.version }}；
          <span v-for="(c, i) in testResult.counts" :key="i" style="margin-right: 12px">{{ i }}: {{ c }}</span>
        </template>
        <template v-else>{{ testResult.error }}</template>
      </el-alert>
    </el-card>

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>导入数据</span>
          <div>
            <el-button type="danger" plain size="small" :disabled="hasRunning" @click="clearData">清空数据</el-button>
            <el-button type="primary" :loading="importing || hasRunning" :disabled="hasRunning" @click="startImport">
              {{ hasRunning ? '导入中…' : '开始导入' }}
            </el-button>
          </div>
        </div>
      </template>
      <el-form :model="importForm" label-width="100px" inline>
        <el-form-item label="索引范围">
          <el-select v-model="importForm.indices" multiple placeholder="留空则用全部配置索引" style="width: 320px">
            <el-option v-for="i in config.indices" :key="i" :label="i" :value="i" />
          </el-select>
        </el-form-item>
        <el-form-item label="增量更新">
          <el-switch v-model="importForm.incremental" />
        </el-form-item>
        <el-form-item label="最大条数(0=不限)">
          <el-input-number v-model="importForm.max_docs" :min="0" :step="10000" />
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>词典聚合</span>
          <el-button type="warning" :loading="lexiconRefreshing" @click="refreshLexicon">从 ES 聚合四词典</el-button>
        </div>
      </template>
      <div style="color: #909399; font-size: 13px">
        将 intentBrand / intentCategory / intentProduct 聚合去重后合并到本地词典（brand/category/product/model）。
      </div>
      <el-alert v-if="lexiconResult" type="success" :closable="false" style="margin-top: 8px">
        聚合完成：{{ JSON.stringify(lexiconResult) }}
      </el-alert>
    </el-card>

    <el-card shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>导入任务</span>
          <el-button size="small" @click="loadTasks">刷新</el-button>
        </div>
      </template>
      <el-alert v-if="hasRunning" type="warning" :closable="false" show-icon style="margin-bottom: 12px">
        正在导入数据，已导入 {{ runningTask.total || 0 }} 条，请稍候…
      </el-alert>
      <el-table :data="tasks" size="small">
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ statusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="total" label="已导入" width="110" />
        <el-table-column prop="message" label="信息" />
        <el-table-column prop="created_at" label="创建时间" width="170" />
      </el-table>
      <el-empty v-if="!tasks.length" description="暂无导入任务" :image-size="60" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const config = reactive({ hosts: [], indices: [], scroll_size: 1000, slices: 4, only_core: true })
const hostInput = ref('')
const indicesInput = ref('')
const testResult = ref(null)
const testing = ref(false)
const importing = ref(false)
const lexiconRefreshing = ref(false)
const lexiconResult = ref(null)

const importForm = reactive({ indices: [], only_core: true, incremental: true, max_docs: 0 })
const tasks = ref([])
let timer = null

const hasRunning = computed(() => tasks.value.some((t) => t.status === 'running'))
const runningTask = computed(() => tasks.value.find((t) => t.status === 'running') || {})

function statusType(s) {
  return { pending: 'info', running: 'warning', done: 'success', failed: 'danger' }[s] || 'info'
}
function statusText(s) {
  return { pending: '待处理', running: '进行中', done: '完成', failed: '失败' }[s] || s
}

async function loadConfig() {
  const c = await api.get('/es/config')
  Object.assign(config, c)
  hostInput.value = (c.hosts || []).join(',')
  indicesInput.value = (c.indices || []).join(',')
}

function parseConfig() {
  config.hosts = hostInput.value.split(',').map((s) => s.trim()).filter(Boolean)
  config.indices = indicesInput.value.split(',').map((s) => s.trim()).filter(Boolean)
}

async function saveConfig() {
  parseConfig()
  await api.put('/es/config', config)
  ElMessage.success('配置已保存')
}

async function testConn() {
  parseConfig()
  await saveConfig()
  testing.value = true
  try {
    testResult.value = await api.post('/es/test')
  } finally {
    testing.value = false
  }
}

async function startImport() {
  importing.value = true
  try {
    const { task_id } = await api.post('/es/import', importForm)
    ElMessage.success(`已开始导入，任务 #${task_id} 运行中，请查看下方任务列表`)
    await loadTasks()
    startPolling()
  } catch (e) {
    ElMessage.error(e.message || '导入失败')
  } finally {
    importing.value = false
  }
}

async function refreshLexicon() {
  lexiconRefreshing.value = true
  lexiconResult.value = null
  try {
    const r = await api.post('/es/lexicon/refresh')
    lexiconResult.value = r.result
    ElMessage.success('词典聚合完成')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    lexiconRefreshing.value = false
  }
}

async function clearData() {
  try {
    await ElMessageBox.confirm(
      '确定清空已导入的全部数据与增量位点吗？清空后需重新「开始导入」。',
      '清空确认',
      { type: 'warning', confirmButtonText: '清空', cancelButtonText: '取消' }
    )
    await api.post('/es/clear')
    ElMessage.success('已清空')
    loadTasks()
  } catch (e) {
    /* 用户取消 */
  }
}

let lastRunning = false
async function loadTasks() {
  tasks.value = await api.get('/es/import/tasks')
  const running = tasks.value.some((t) => t.status === 'running')
  if (lastRunning && !running) {
    const done = tasks.value.find((t) => t.status === 'done')
    ElMessage.success(done ? `导入完成，共 ${done.total} 条` : '导入结束')
  }
  lastRunning = running
}

function startPolling() {
  loadTasks()
  if (timer) return
  timer = setInterval(loadTasks, 3000)
}

onMounted(() => {
  loadConfig()
  loadTasks()
})
onUnmounted(() => timer && clearInterval(timer))
</script>
