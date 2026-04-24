<script setup lang="ts">
import { onMounted } from 'vue'
import { useMetricsStore } from '../stores/metrics'
import SummaryCards from '../components/SummaryCards.vue'
import ConfusionMatrixHeatmap from '../components/ConfusionMatrixHeatmap.vue'
import PerClassBarChart from '../components/PerClassBarChart.vue'

const store = useMetricsStore()

onMounted(() => store.load())
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-baseline justify-between">
      <h2 class="text-2xl font-bold text-slate-900">Training Results</h2>
      <button
        class="text-sm text-indigo-600 hover:text-indigo-800"
        @click="store.load(true)"
      >
        Refresh
      </button>
    </div>

    <div v-if="store.loading" class="text-slate-500">Loading metrics…</div>
    <div v-else-if="store.error" class="p-4 bg-rose-50 border border-rose-200 rounded text-rose-700">
      {{ store.error }}
    </div>

    <template v-else-if="store.summary && store.confusion && store.perClass">
      <SummaryCards :summary="store.summary" />
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ConfusionMatrixHeatmap :data="store.confusion" />
        <PerClassBarChart :data="store.perClass" />
      </div>
    </template>
  </div>
</template>
