<template>
  <div>
    <h2 style="margin-top: 0">在线试测</h2>

    <el-card shadow="never" style="margin-bottom: 16px">
      <el-form label-width="60px">
        <el-form-item label="标题">
          <el-input v-model="title" placeholder="标题（可选）" style="max-width: 600px" />
        </el-form-item>
        <el-form-item label="正文">
          <el-input
            v-model="note"
            type="textarea"
            :rows="3"
            placeholder="正文（笔记描述），例如：iPhone 17 开箱/验机/配件分享"
          />
        </el-form-item>
        <el-form-item label="评论">
          <el-input
            v-model="comment"
            type="textarea"
            :rows="3"
            placeholder="评论，例如：壳有吗 / 我想买一台，有优惠吗"
          />
        </el-form-item>
      </el-form>
      <div style="margin-top: 12px">
        <el-button type="primary" :loading="loading" @click="doPredict">识别</el-button>
        <el-button @click="examples">示例</el-button>
      </div>
      <div style="margin-top: 8px; color: #909399; font-size: 12px">
        模型会结合「正文 + 评论」联合判断：实体可能来自正文或评论，{{ intentName }}意图主要结合评论与正文上下文判定。
      </div>
    </el-card>

    <el-alert v-if="result.errors && result.errors.length" type="warning" :closable="false" style="margin-bottom: 16px">
      <div v-for="e in result.errors" :key="e">{{ e }}</div>
    </el-alert>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card shadow="never" header="实体识别结果">
          <el-table :data="result.entities" size="small">
            <el-table-column prop="text" label="实体" />
            <el-table-column label="类型" width="90">
              <template #default="{ row }">
                <el-tag size="small" :type="typeColor(row.type)">{{ typeLabel(row.type) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="来源" width="80">
              <template #default="{ row }">
                <el-tag size="small" effect="plain" type="info">{{ row.source || '正文' }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="置信度" width="90">
              <template #default="{ row }">
                <span v-if="row.prob !== undefined">{{ Math.round(row.prob * 100) }}%</span>
                <span v-else>-</span>
              </template>
            </el-table-column>
          </el-table>
          <el-empty v-if="!result.entities.length" description="未识别到实体" :image-size="60" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="never" header="意图打分结果">
          <div v-for="it in result.intents" :key="it.name" style="margin-bottom: 12px">
            <div style="font-size: 14px; margin-bottom: 6px">
              意图「{{ it.name }}」：
              <span style="font-weight: 600; font-size: 22px; color: #409eff">{{ it.score }} 分</span>
              <el-tag size="small" style="margin-left: 8px">{{ it.level }}</el-tag>
            </div>
            <el-progress :percentage="Math.round(it.prob * 100)" :stroke-width="14" :color="scoreColor(it.score)" />
          </div>
          <el-empty v-if="!result.intents.length" description="意图模型未就绪" :image-size="60" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const title = ref('')
const note = ref('')
const comment = ref('')
const loading = ref(false)
const result = ref({ entities: [], intents: [], errors: [] })
const intentName = ref('购物')

onMounted(async () => {
  try {
    const cfg = await api.get('/intent/config')
    if (cfg && cfg.name) intentName.value = cfg.name
  } catch (e) {
    // 忽略，保留默认「购物」
  }
})

const TYPE_MAP = { BRAND: '品牌', CATEGORY: '品类', PRODUCT: '商品', MODEL: '型号' }
function typeLabel(t) { return TYPE_MAP[t] || t }
function typeColor(t) {
  return { BRAND: '', CATEGORY: 'success', PRODUCT: 'warning', MODEL: 'info' }[t] || ''
}
function scoreColor(s) {
  if (s <= 2) return '#909399'
  if (s === 3) return '#e6a23c'
  if (s === 4) return '#f56c6c'
  return '#f56c6c'
}

async function doPredict() {
  if (!note.value.trim() && !comment.value.trim()) {
    ElMessage.warning('请至少输入正文或评论')
    return
  }
  loading.value = true
  try {
    result.value = await api.post('/predict', {
      title: title.value,
      note: note.value,
      comment: comment.value,
    })
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    loading.value = false
  }
}

function examples() {
  title.value = ''
  note.value = '把iPhone17当主力机的第67天，如果你正打算买iPhone 17，建议先看这67天的真实感受'
  comment.value = '我现在用红米turbo3想换17，平常拍照微信抖音刷题，能换吧'
}
</script>
