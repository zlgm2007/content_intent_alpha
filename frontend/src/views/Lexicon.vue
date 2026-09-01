<template>
  <div>
    <h2 style="margin-top: 0">词典管理</h2>

    <el-card shadow="never">
      <el-tabs v-model="activeType">
        <el-tab-pane v-for="t in types" :key="t" :name="t">
          <template #label>
            {{ t }} <span style="color: #909399">({{ (lexicon[t] || []).length }})</span>
          </template>
        </el-tab-pane>
      </el-tabs>

      <div style="margin-bottom: 16px">
        <el-input
          v-model="newWords"
          type="textarea"
          :rows="2"
          placeholder="批量添加，逗号/空格/换行分隔，如：小米, 华为"
          style="margin-bottom: 8px"
        />
        <el-button type="primary" @click="addWords">添加词条</el-button>
      </div>

      <div v-if="words.length" style="max-height: 420px; overflow-y: auto; border: 1px solid #ebeef5; border-radius: 8px; padding: 12px">
        <el-tag
          v-for="w in words"
          :key="w"
          closable
          style="margin: 4px"
          @close="removeWord(w)"
        >{{ w }}</el-tag>
      </div>
      <el-empty v-else description="暂无词条，可从 ES 聚合或手动添加" :image-size="60" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const types = ['brand', 'category', 'product', 'model']
const activeType = ref('brand')
const lexicon = ref({})
const newWords = ref('')

const words = computed(() => lexicon.value[activeType.value] || [])

async function load() {
  lexicon.value = await api.get('/lexicon')
}

async function addWords() {
  const list = newWords.value
    .split(/[,，\s\n]+/)
    .map((s) => s.trim())
    .filter(Boolean)
  if (!list.length) return
  await api.post(`/lexicon/${activeType.value}`, { words: list })
  newWords.value = ''
  ElMessage.success(`已添加 ${list.length} 条`)
  load()
}

async function removeWord(word) {
  await api.delete(`/lexicon/${activeType.value}`, { params: { word } })
  load()
}

onMounted(load)
</script>
