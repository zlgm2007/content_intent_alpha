<template>
  <div>
    <h2 style="margin-top: 0">数据准备</h2>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header><span>NER 训练数据（实体识别）</span></template>
          <div style="color: #909399; font-size: 13px; margin-bottom: 12px">
            从已导入样本的「正文 + 评论」中，把意图品牌/品类/商品/型号对齐为 BIO 标注（实体可能来自正文或评论）。
          </div>
          <el-form label-width="100px" inline>
            <el-form-item label="样本上限">
              <el-input-number v-model="nerForm.limit" :min="0" :step="5000" />
            </el-form-item>
            <el-form-item label="最少实体数">
              <el-input-number v-model="nerForm.min_entities" :min="1" :max="4" />
            </el-form-item>
            <el-form-item label="每模式条数">
              <el-input-number v-model="nerForm.max_per_pattern" :min="0" :max="50" />
            </el-form-item>
            <el-form-item label="标注方式">
              <el-select v-model="nerForm.annotator" style="width: 130px">
                <el-option label="词典全量(dict)" value="dict" />
                <el-option label="ES真值对齐(gold)" value="gold" />
              </el-select>
            </el-form-item>
            <el-form-item label="留出比例">
              <el-input-number v-model="nerForm.holdout_ratio" :min="0" :max="0.5" :step="0.05" :precision="2" />
            </el-form-item>
          </el-form>
          <div style="color: #909399; font-size: 12px; margin-bottom: 10px">
            每种「实体组合」最多保留 {{ nerForm.max_per_pattern || '∞' }} 条（0=全量）。全量 13 万条只有 3.2 万种标注模式，
            设 5 可压到 5.4 万条、训练从 52 分钟降至 22 分钟，实体词覆盖率仍为 100%。
          </div>
          <div v-if="nerForm.annotator === 'gold'" style="color: #E6A23C; font-size: 12px; margin-bottom: 8px">
            ⚠️ gold 方式只标 ES 真值字段里出现的词，实测漏标 51.1%（正文出现了但真值没写的实体一个都不标），
            模型会学到「该词可标可不标」并产生碎片实体。一般情况请用「词典全量」。
          </div>
          <el-button type="primary" :loading="nerLoading" @click="prepareNer">生成 NER 数据</el-button>
          <el-alert v-if="nerResult" type="success" :closable="false" style="margin-top: 8px">
            已生成 {{ nerResult.samples }} 条标注样本 → {{ nerResult.out }}
            <div v-if="nerResult.dedup_dropped" style="font-size: 12px">
              按标注模式去重丢弃 {{ nerResult.dedup_dropped }} 条冗余样本，覆盖 {{ nerResult.patterns }} 种标注模式
            </div>
            <div v-if="nerResult.holdout" style="font-size: 12px; margin-top: 4px">
              留出集 {{ nerResult.holdout }} 条 / {{ nerResult.holdout_entities }} 个实体 → {{ nerResult.holdout_out }}
            </div>
            <div style="font-size: 12px">标注方式：{{ nerResult.annotator }}，词典映射 {{ nerResult.type_map_size }} 词</div>
          </el-alert>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="never">
          <template #header><span>意图训练数据（0–5 分打分）</span></template>
          <div style="color: #909399; font-size: 13px; margin-bottom: 12px">
            用「正文 + 评论」+ intentScore 生成 6 分类数据（评论结合正文上下文判断{{ intentName }}意图），无分样本按 0 分处理，支持采样平衡。
          </div>
          <el-form label-width="110px" inline>
            <el-form-item label="样本上限">
              <el-input-number v-model="intentForm.limit" :min="0" :step="10000" />
            </el-form-item>
            <el-form-item label="采样平衡">
              <el-switch v-model="intentForm.balance" />
            </el-form-item>
            <el-form-item label="每类上限">
              <el-input-number v-model="intentForm.max_per_class" :min="0" :step="5000" />
            </el-form-item>
            <el-form-item label="留出比例">
              <el-input-number v-model="intentForm.holdout_ratio" :min="0" :max="0.5" :step="0.05" :precision="2" />
            </el-form-item>
          </el-form>
          <div style="color: #E6A23C; font-size: 12px; margin-bottom: 8px">
            留出集在「去重之后、采样平衡之前」随机预留，保持真实分布，用于评估模型泛化能力。
            若先平衡再留出，0/2/4 这类小类会被 100% 纳入训练、留出集里一条都不剩，评估结论不可信。
          </div>
          <el-button type="primary" :loading="intentLoading" @click="prepareIntent">生成意图数据</el-button>
          <el-alert v-if="intentResult" type="success" :closable="false" style="margin-top: 8px">
            共 {{ intentResult.samples }} 条<template v-if="intentResult.dups_dropped">（去重丢弃 {{ intentResult.dups_dropped }} 条重复文本）</template>。
            <div style="font-size: 12px">分布(采样前→后)：{{ fmtDist(intentResult.dist_before) }} → {{ fmtDist(intentResult.dist_after) }}</div>
            <template v-if="intentResult.holdout">
              <div style="font-size: 12px; margin-top: 4px">
                留出集 {{ intentResult.holdout }} 条 → {{ intentResult.holdout_out }}
              </div>
              <div style="font-size: 12px">留出集分布：{{ fmtDist(intentResult.holdout_dist) }}</div>
              <div v-if="!holdoutComplete" style="font-size: 12px; color: #F56C6C">
                ⚠️ 留出集缺少部分分值类别，六分类已退化，泛化评估结论不可采信。
              </div>
            </template>
          </el-alert>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const nerForm = reactive({
  limit: 0, min_entities: 1, max_per_pattern: 5,
  annotator: 'dict', holdout_ratio: 0.1,
})
const intentForm = reactive({ limit: 0, balance: true, max_per_class: 20000, holdout_ratio: 0.1 })
const nerLoading = ref(false)
const intentLoading = ref(false)
const nerResult = ref(null)
const intentResult = ref(null)
const intentName = ref('购物')

onMounted(async () => {
  try {
    const cfg = await api.get('/intent/config')
    if (cfg && cfg.name) intentName.value = cfg.name
  } catch (e) {
    // 忽略，保留默认「购物」
  }
})

function fmtDist(d) {
  if (!d) return ''
  return Object.entries(d).map(([k, v]) => `${k}分:${v}`).join(', ')
}

// 留出集必须 0–5 六类齐全，缺任何一类都说明评估已退化
const holdoutComplete = computed(() => {
  const d = intentResult.value && intentResult.value.holdout_dist
  if (!d) return false
  return [0, 1, 2, 3, 4, 5].every(k => (d[k] || d[String(k)] || 0) > 0)
})

async function prepareNer() {
  nerLoading.value = true
  nerResult.value = null
  try {
    nerResult.value = await api.post('/data/ner/prepare', nerForm)
    ElMessage.success('NER 数据已生成')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    nerLoading.value = false
  }
}

async function prepareIntent() {
  intentLoading.value = true
  intentResult.value = null
  try {
    intentResult.value = await api.post('/data/intent/prepare', intentForm)
    ElMessage.success('意图数据已生成')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    intentLoading.value = false
  }
}
</script>
