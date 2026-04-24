import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/dashboard' },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: () => import('../views/DashboardView.vue'),
  },
  {
    path: '/predict',
    name: 'predict',
    component: () => import('../views/PredictView.vue'),
  },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
