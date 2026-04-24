<script setup lang="ts">
import { computed } from 'vue'
import type { SummaryMetrics } from '../types'

const props = defineProps<{ summary: SummaryMetrics }>()

const cards = computed(() => [
  { label: 'Accuracy', value: props.summary.accuracy, pct: true, tone: 'indigo' },
  { label: 'Macro F1', value: props.summary.macro_f1, pct: true, tone: 'emerald' },
  { label: 'Weighted F1', value: props.summary.weighted_f1, pct: true, tone: 'sky' },
  { label: 'Test Samples', value: props.summary.total_samples, pct: false, tone: 'slate' },
])

const toneBg: Record<string, string> = {
  indigo: 'bg-indigo-50 text-indigo-700',
  emerald: 'bg-emerald-50 text-emerald-700',
  sky: 'bg-sky-50 text-sky-700',
  slate: 'bg-slate-100 text-slate-700',
}

function format(value: number, pct: boolean) {
  return pct ? `${(value * 100).toFixed(2)}%` : value.toLocaleString()
}
</script>

<template>
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
    <div
      v-for="card in cards"
      :key="card.label"
      class="bg-white rounded-lg border border-slate-200 shadow-sm p-5"
    >
      <div
        class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
        :class="toneBg[card.tone]"
      >
        {{ card.label }}
      </div>
      <div class="mt-3 text-3xl font-bold text-slate-900">
        {{ format(card.value, card.pct) }}
      </div>
    </div>
  </div>
</template>
