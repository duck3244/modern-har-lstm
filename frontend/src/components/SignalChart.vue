<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ signal: number[][] }>()

const CHANNEL_NAMES = [
  'body_acc_x', 'body_acc_y', 'body_acc_z',
  'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
  'total_acc_x', 'total_acc_y', 'total_acc_z',
]

const COLORS = [
  '#6366f1', '#10b981', '#0ea5e9',
  '#f59e0b', '#ef4444', '#8b5cf6',
  '#14b8a6', '#ec4899', '#22c55e',
]

const option = computed(() => {
  const T = props.signal.length
  const nChannels = CHANNEL_NAMES.length
  const xAxis = Array.from({ length: T }, (_, i) => i)

  // 3x3 grid of small multiples.
  const grids = []
  const xAxes = []
  const yAxes = []
  const series = []
  const titles = []

  const cols = 3
  const rowGap = 8
  const colGap = 6
  const topPad = 6
  const rowH = (100 - topPad - rowGap * 2) / 3
  const colW = (100 - colGap * 2) / 3

  for (let c = 0; c < nChannels; c++) {
    const col = c % cols
    const row = Math.floor(c / cols)
    const left = col * (colW + colGap) + colGap / 2
    const top = topPad + row * (rowH + rowGap)

    grids.push({
      left: `${left + 3}%`,
      top: `${top + 3}%`,
      width: `${colW - 3}%`,
      height: `${rowH - 6}%`,
      containLabel: true,
    })
    xAxes.push({
      gridIndex: c,
      type: 'category',
      data: xAxis,
      axisLabel: { fontSize: 9 },
      splitLine: { show: false },
    })
    yAxes.push({
      gridIndex: c,
      type: 'value',
      axisLabel: { fontSize: 9, formatter: (v: number) => v.toFixed(1) },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
    })
    series.push({
      type: 'line',
      showSymbol: false,
      smooth: false,
      lineStyle: { width: 1.2, color: COLORS[c] },
      xAxisIndex: c,
      yAxisIndex: c,
      data: props.signal.map((row) => row[c]),
    })
    titles.push({
      text: CHANNEL_NAMES[c],
      textStyle: { fontSize: 11, fontWeight: 600, color: '#334155' },
      left: `${left + 3}%`,
      top: `${top + 0.5}%`,
    })
  }

  return {
    animation: false,
    title: titles,
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series,
    tooltip: { trigger: 'axis', axisPointer: { type: 'line' } },
  }
})
</script>

<template>
  <div class="bg-white rounded-lg border border-slate-200 shadow-sm p-5">
    <h3 class="text-lg font-semibold text-slate-800 mb-4">Signal Channels</h3>
    <v-chart :option="option" :style="{ height: '540px' }" autoresize />
  </div>
</template>
