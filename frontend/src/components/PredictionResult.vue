<script setup lang="ts">
import { computed } from 'vue'
import type { PredictResponse } from '../types'

const props = defineProps<{
  result: PredictResponse
  trueLabel?: string | null
}>()

const isCorrect = computed(() => {
  return props.trueLabel ? props.trueLabel === props.result.predicted_label : null
})

const option = computed(() => {
  const entries = Object.entries(props.result.probabilities)
  const labels = entries.map(([k]) => k)
  const values = entries.map(([, v]) => v)
  const topIdx = values.indexOf(Math.max(...values))
  return {
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v: number) => (v * 100).toFixed(2) + '%',
    },
    grid: { top: 20, left: 120, right: 30, bottom: 30 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: { formatter: (v: number) => (v * 100).toFixed(0) + '%' },
    },
    yAxis: { type: 'category', data: labels },
    series: [
      {
        type: 'bar',
        data: values.map((v, i) => ({
          value: v,
          itemStyle: { color: i === topIdx ? '#6366f1' : '#cbd5e1' },
        })),
        label: {
          show: true,
          position: 'right',
          formatter: (p: { value: number }) => (p.value * 100).toFixed(1) + '%',
        },
      },
    ],
  }
})
</script>

<template>
  <div class="bg-white rounded-lg border border-slate-200 shadow-sm p-5">
    <h3 class="text-lg font-semibold text-slate-800 mb-4">Prediction</h3>
    <div class="flex items-center gap-4 mb-4">
      <div>
        <div class="text-sm text-slate-500">Predicted</div>
        <div class="text-2xl font-bold text-indigo-700">
          {{ result.predicted_label }}
        </div>
        <div class="text-sm text-slate-500">
          confidence {{ (result.confidence * 100).toFixed(2) }}%
        </div>
      </div>
      <div v-if="trueLabel" class="pl-6 border-l border-slate-200">
        <div class="text-sm text-slate-500">True</div>
        <div class="text-2xl font-bold text-slate-800">{{ trueLabel }}</div>
        <div
          class="text-sm font-medium"
          :class="isCorrect ? 'text-emerald-600' : 'text-rose-600'"
        >
          {{ isCorrect ? '✓ Match' : '✗ Mismatch' }}
        </div>
      </div>
    </div>
    <v-chart :option="option" :style="{ height: '260px' }" autoresize />
  </div>
</template>
