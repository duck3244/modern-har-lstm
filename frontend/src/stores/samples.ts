import { defineStore } from 'pinia'
import { ref } from 'vue'
import { fetchSamplesInfo } from '../api/endpoints'
import type { SamplesInfo } from '../types'

export const useSamplesStore = defineStore('samples', () => {
  const info = ref<SamplesInfo | null>(null)

  async function load() {
    if (info.value) return
    info.value = await fetchSamplesInfo()
  }

  return { info, load }
})
