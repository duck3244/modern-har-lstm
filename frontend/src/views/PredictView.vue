<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  fetchRandomSample,
  fetchSample,
  predictSignal,
  predictUpload,
} from '../api/endpoints'
import { useSamplesStore } from '../stores/samples'
import type { PredictResponse, SampleResponse } from '../types'
import SignalChart from '../components/SignalChart.vue'
import PredictionResult from '../components/PredictionResult.vue'

const samplesStore = useSamplesStore()

const idxInput = ref<number>(0)
const sample = ref<SampleResponse | null>(null)
const prediction = ref<PredictResponse | null>(null)
const sourceLabel = ref<string | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

onMounted(() => samplesStore.load())

async function loadByIndex() {
  const total = samplesStore.info?.total ?? 0
  if (idxInput.value < 0 || idxInput.value >= total) {
    error.value = `Index must be in [0, ${total - 1}]`
    return
  }
  error.value = null
  loading.value = true
  try {
    sample.value = await fetchSample(idxInput.value)
    prediction.value = await predictSignal(sample.value.signal)
    sourceLabel.value = `Test sample #${sample.value.index}`
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    loading.value = false
  }
}

async function loadRandom() {
  error.value = null
  loading.value = true
  try {
    sample.value = await fetchRandomSample()
    idxInput.value = sample.value.index
    prediction.value = await predictSignal(sample.value.signal)
    sourceLabel.value = `Random test sample #${sample.value.index}`
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    loading.value = false
  }
}

async function onFileSelected(event: Event) {
  const files = (event.target as HTMLInputElement).files
  if (!files || !files.length) return
  const file = files[0]
  error.value = null
  loading.value = true
  try {
    prediction.value = await predictUpload(file)
    sample.value = null
    sourceLabel.value = `Uploaded: ${file.name}`
  } catch (e) {
    error.value = errorMessage(e)
  } finally {
    loading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

function errorMessage(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    // axios error
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const detail = (e as any).response?.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ')
  }
  return e instanceof Error ? e.message : 'Prediction failed'
}
</script>

<template>
  <div class="space-y-6">
    <h2 class="text-2xl font-bold text-slate-900">Predict Activity</h2>

    <div class="bg-white rounded-lg border border-slate-200 shadow-sm p-5">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Test sample index
            <span class="text-slate-400">
              (0 – {{ (samplesStore.info?.total ?? 1) - 1 }})
            </span>
          </label>
          <div class="flex gap-2">
            <input
              v-model.number="idxInput"
              type="number"
              :min="0"
              :max="(samplesStore.info?.total ?? 1) - 1"
              class="w-full rounded-md border-slate-300 shadow-sm px-3 py-2 text-sm focus:ring-indigo-500 focus:border-indigo-500"
            />
            <button
              class="px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium disabled:opacity-50"
              :disabled="loading"
              @click="loadByIndex"
            >
              Predict
            </button>
          </div>
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Random sample</label>
          <button
            class="w-full px-4 py-2 rounded-md bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium disabled:opacity-50"
            :disabled="loading"
            @click="loadRandom"
          >
            Pick Random & Predict
          </button>
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">
            Upload file (CSV or JSON, shape 128×9)
          </label>
          <input
            ref="fileInput"
            type="file"
            accept=".csv,.json,.txt"
            class="block w-full text-sm text-slate-600 file:mr-3 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-slate-100 file:text-slate-700 hover:file:bg-slate-200"
            :disabled="loading"
            @change="onFileSelected"
          />
        </div>
      </div>

      <p v-if="sourceLabel" class="mt-4 text-sm text-slate-500">
        Source: <span class="font-medium text-slate-700">{{ sourceLabel }}</span>
      </p>
      <p v-if="error" class="mt-4 text-sm text-rose-600">{{ error }}</p>
    </div>

    <div v-if="loading" class="text-slate-500">Running inference…</div>

    <div v-if="prediction" class="grid grid-cols-1 gap-6">
      <PredictionResult
        :result="prediction"
        :true-label="sample?.true_label ?? null"
      />
      <SignalChart v-if="sample" :signal="sample.signal" />
    </div>
  </div>
</template>
