<template>
  <div>
    <h2 style="margin-top: 0">模型管理</h2>

    <el-row :gutter="16">
      <el-col :span="12" v-for="m in modelCards" :key="m.type">
        <el-card shadow="never">
          <template #header><span>{{ m.label }}</span></template>
          <div style="margin-bottom: 8px">
            <el-tag :type="m.exists ? 'success' : 'info'" size="small">
              {{ m.exists ? '已导出' : '未导出' }}
            </el-tag>
            <span v-if="m.exists" style="margin-left: 8px; color: #909399; font-size: 12px">{{ m.size_mb }} MB</span>
          </div>
          <el-button type="primary" size="small" :loading="m.exporting" @click="exportModel(m)">导出 ONNX</el-button>
          <el-button size="small" :disabled="!m.exists" @click="download(m)">下载</el-button>
          <div style="color: #909399; font-size: 12px; margin-top: 8px">{{ m.hint }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" style="margin-top: 16px">
      <template #header><span>导出记录</span></template>
      <el-table :data="versions" size="small">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="model_type" label="模型" width="90" />
        <el-table-column prop="path" label="路径" />
        <el-table-column prop="created_at" label="时间" width="170" />
      </el-table>
      <el-empty v-if="!versions.length" description="暂无导出记录" :image-size="60" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const cards = ref([
  { type: 'ner', label: 'NER 实体识别', hint: '需先在训练中心训练 NER 模型' },
  { type: 'intent', label: '意图打分', hint: '需先在训练中心训练意图模型' },
])
const versions = ref([])

const modelCards = ref([])
async function load() {
  const data = await api.get('/model')
  versions.value = data.versions || []
  modelCards.value = cards.value.map((c) => ({
    ...c,
    exists: data[c.type]?.exists,
    size_mb: data[c.type]?.size_mb || 0,
    exporting: false,
  }))
}

async function exportModel(m) {
  m.exporting = true
  try {
    await api.post(`/model/export/${m.type}`)
    ElMessage.success(`${m.label} ONNX 已导出`)
    load()
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    m.exporting = false
  }
}

function download(m) {
  window.open(`/api/model/download/${m.type}`, '_blank')
}

onMounted(load)
</script>
