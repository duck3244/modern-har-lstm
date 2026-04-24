<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ConfusionMatrixResponse } from '../types'

const props = defineProps<{ data: ConfusionMatrixResponse }>()

const mode = ref<'normalized' | 'counts'>('normalized')

const option = computed(() => {
  const { labels, matrix, normalized } = props.data
  const source = mode.value === 'normalized' ? normalized : matrix

  // ECharts heatmap expects [xIdx, yIdx, value]. Origin bottom-left → flip rows.
  const cells: [number, number, number][] = []
  for (let i = 0; i < source.length; i++) {
    for (let j = 0; j < source[i].length; j++) {
      cells.push([j, labels.length - 1 - i, source[i][j]])
    }
  }

  const max =
    mode.value === 'normalized'
      ? 1
      : Math.max(...matrix.flat())

  return {
    tooltip: {
      position: 'top',
      formatter: (params: { value: [number, number, number] }) => {
        const [x, y, v] = params.value
        const pred = labels[x]
        const truth = labels[labels.length - 1 - y]
        const display =
          mode.value === 'normalized'
            ? `${(v * 100).toFixed(1)}%`
            : v.toString()
        return `True: <b>${truth}</b><br/>Pred: <b>${pred}</b><br/>${display}`
      },
    },
    grid: { top: 60, left: 140, right: 40, bottom: 80 },
    xAxis: {
      type: 'category',
      data: labels,
      name: 'Predicted',
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: { rotate: 30 },
      splitArea: { show: true },
    },
    yAxis: {
      type: 'category',
      data: [...labels].reverse(),
      name: 'True',
      nameLocation: 'middle',
      nameGap: 110,
      splitArea: { show: true },
    },
    visualMap: {
      min: 0,
      max,
      calculable: true,
      orient: 'vertical',
      right: 0,
      top: 'middle',
      inRange: { color: ['#eef2ff', '#6366f1', '#312e81'] },
    },
    series: [
      {
        type: 'heatmap',
        data: cells,
        label: {
          show: true,
          formatter: (p: { value: [number, number, number] }) => {
            const v = p.value[2]
            return mode.value === 'normalized'
              ? (v * 100).toFixed(0) + '%'
              : v.toString()
          },
          color: '#1e293b',
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 8,
            shadowColor: 'rgba(0,0,0,0.25)',
          },
        },
      },
    ],
  }
})
</script>

<template>
  <div class="bg-white rounded-lg border border-slate-200 shadow-sm p-5">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-slate-800">Confusion Matrix</h3>
      <div class="inline-flex bg-slate-100 rounded-md p-1">
        <button
          class="px-3 py-1 text-sm rounded"
          :class="mode === 'normalized' ? 'bg-white shadow text-indigo-700' : 'text-slate-600'"
          @click="mode = 'normalized'"
        >
          Normalized
        </button>
        <button
          class="px-3 py-1 text-sm rounded"
          :class="mode === 'counts' ? 'bg-white shadow text-indigo-700' : 'text-slate-600'"
          @click="mode = 'counts'"
        >
          Counts
        </button>
      </div>
    </div>
    <v-chart :option="option" :style="{ height: '480px' }" autoresize />
  </div>
</template>
