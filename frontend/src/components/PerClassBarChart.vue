<script setup lang="ts">
import { computed } from 'vue'
import type { PerClassResponse } from '../types'

const props = defineProps<{ data: PerClassResponse }>()

const option = computed(() => {
  const labels = props.data.rows.map((r) => r.label)
  return {
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v: number) => (v * 100).toFixed(1) + '%',
    },
    legend: { data: ['Precision', 'Recall', 'F1'], top: 0 },
    grid: { top: 50, left: 60, right: 20, bottom: 80 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { rotate: 30 },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' },
    },
    series: [
      {
        name: 'Precision',
        type: 'bar',
        data: props.data.rows.map((r) => r.precision),
        itemStyle: { color: '#6366f1' },
      },
      {
        name: 'Recall',
        type: 'bar',
        data: props.data.rows.map((r) => r.recall),
        itemStyle: { color: '#10b981' },
      },
      {
        name: 'F1',
        type: 'bar',
        data: props.data.rows.map((r) => r.f1),
        itemStyle: { color: '#0ea5e9' },
      },
    ],
  }
})
</script>

<template>
  <div class="bg-white rounded-lg border border-slate-200 shadow-sm p-5">
    <h3 class="text-lg font-semibold text-slate-800 mb-4">Per-Class Metrics</h3>
    <v-chart :option="option" :style="{ height: '360px' }" autoresize />
  </div>
</template>
