<template>
  <el-container style="height: 100vh">
    <el-aside width="220px" style="border-right: 1px solid #e4e7ed">
      <div style="padding: 20px 16px; font-size: 16px; font-weight: 600">内容意图识别 · 管理台</div>
      <el-menu :default-active="$route.path" router>
        <el-menu-item index="/">
          <el-icon><DataBoard /></el-icon>
          <span>总览</span>
        </el-menu-item>
        <el-menu-item index="/es-import">
          <el-icon><Download /></el-icon>
          <span>ES 数据导入</span>
        </el-menu-item>
        <el-menu-item index="/lexicon">
          <el-icon><Collection /></el-icon>
          <span>词典管理</span>
        </el-menu-item>
        <el-menu-item index="/data-prep">
          <el-icon><Files /></el-icon>
          <span>数据准备</span>
        </el-menu-item>
        <el-menu-item index="/train">
          <el-icon><Cpu /></el-icon>
          <span>训练中心</span>
        </el-menu-item>
        <el-menu-item index="/model">
          <el-icon><Box /></el-icon>
          <span>模型管理</span>
        </el-menu-item>
        <el-menu-item index="/test">
          <el-icon><Promotion /></el-icon>
          <span>在线试测</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-main style="background: #f5f7fa; padding: 20px">
      <router-view />
    </el-main>
  </el-container>
</template>

<script setup>
import { DataBoard, Download, Collection, Files, Cpu, Box, Promotion } from '@element-plus/icons-vue'
</script>
