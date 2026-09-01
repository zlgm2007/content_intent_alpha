import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '../layout/MainLayout.vue'

const routes = [
  {
    path: '/',
    component: MainLayout,
    children: [
      { path: '', name: 'dashboard', component: () => import('../views/Dashboard.vue') },
      { path: 'es-import', name: 'esImport', component: () => import('../views/EsImport.vue') },
      { path: 'lexicon', name: 'lexicon', component: () => import('../views/Lexicon.vue') },
      { path: 'data-prep', name: 'dataPrep', component: () => import('../views/DataPrep.vue') },
      { path: 'train', name: 'train', component: () => import('../views/Train.vue') },
      { path: 'model', name: 'model', component: () => import('../views/Model.vue') },
      { path: 'test', name: 'test', component: () => import('../views/Test.vue') },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
