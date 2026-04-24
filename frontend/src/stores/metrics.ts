import { defineStore } from 'pinia'
import { ref } from 'vue'
import {
  fetchConfusionMatrix,
  fetchPerClass,
  fetchSummary,
} from '../api/endpoints'
import type {
  ConfusionMatrixResponse,
  PerClassResponse,
  SummaryMetrics,
} from '../types'

export const useMetricsStore = defineStore('metrics', () => {
  const summary = ref<SummaryMetrics | null>(null)
  const confusion = ref<ConfusionMatrixResponse | null>(null)
  const perClass = ref<PerClassResponse | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function load(force = false) {
    if (!force && summary.value && confusion.value && perClass.value) return
    loading.value = true
    error.value = null
    try {
      const [s, c, p] = await Promise.all([
        fetchSummary(),
        fetchConfusionMatrix(),
        fetchPerClass(),
      ])
      summary.value = s
      confusion.value = c
      perClass.value = p
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load metrics'
    } finally {
      loading.value = false
    }
  }

  return { summary, confusion, perClass, loading, error, load }
})
