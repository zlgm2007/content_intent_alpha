<template>
  <div>
    <h2 style="margin-top: 0">总览</h2>

    <el-row :gutter="16">
      <el-col :span="6">
        <el-card shadow="never">
          <div class="stat-label">ES 连接</div>
          <div class="stat-value">
            <el-tag :type="health.es_ok ? 'success' : 'danger'">
              {{ health.es_ok ? '正常' : '未连接' }}
            </el-tag>
          </div>
          <div class="stat-sub" v-if="health.es_cluster">{{ health.es_cluster }} v{{ health.es_version }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="never">
          <div class="stat-label">已导入样本</div>
          <div class="stat-value">{{ stats.es_docs }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="never">
          <div class="stat-label">词典总量</div>
          <div class="stat-value">{{ lexiconTotal }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="never">
          <div class="stat-label">最近导入</div>
          <div class="stat-value" style="font-size: 16px">
            {{ recentImport ? recentImport.status : '—' }}
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px">
      <el-col :span="12">
        <el-card shadow="never" header="意图分数分布（已导入样本）">
          <div v-if="stats.score_dist && stats.score_dist.length" style="display: flex; gap: 12px; flex-wrap: wrap">
            <div v-for="d in stats.score_dist" :key="d.score" class="score-pill">
              <span class="score-num">{{ d.score === null ? '无分' : d.score + '分' }}</span>
              <span class="score-cnt">{{ d.count }}</span>
            </div>
          </div>
          <el-empty v-else description="暂无数据，请先到「ES 数据导入」拉取" :image-size="60" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="never" header="词典词条数">
          <el-table :data="lexiconRows" size="small">
            <el-table-column prop="type" label="类型" width="120" />
            <el-table-column prop="count" label="词条数" />
          </el-table>
          <el-empty v-if="!lexiconRows.length" description="暂无词典" :image-size="60" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api'

const health = ref({ es_ok: false })
const stats = ref({ es_docs: 0, score_dist: [], lexicon: {}, recent_imports: [] })

const lexiconTotal = computed(() =>
  Object.values(stats.value.lexicon || {}).reduce((a, b) => a + b, 0)
)
const lexiconRows = computed(() =>
  Object.entries(stats.value.lexicon || {}).map(([type, count]) => ({ type, count }))
)
const recentImport = computed(() => stats.value.recent_imports?.[0])

async function load() {
  const [h, s] = await Promise.all([api.get('/health'), api.get('/stats')])
  health.value = h
  stats.value = s
}

onMounted(load)
</script>

<style scoped>
.stat-label {
  color: #909399;
  font-size: 13px;
}
.stat-value {
  font-size: 28px;
  font-weight: 600;
  margin: 8px 0;
}
.stat-sub {
  color: #909399;
  font-size: 12px;
}
.score-pill {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 8px 14px;
  text-align: center;
  min-width: 64px;
}
.score-num {
  display: block;
  font-size: 14px;
  font-weight: 600;
}
.score-cnt {
  display: block;
  font-size: 12px;
  color: #909399;
}
</style>
