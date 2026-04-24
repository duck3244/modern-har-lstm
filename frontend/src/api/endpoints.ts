import { api } from './client'
import type {
  ConfusionMatrixResponse,
  HealthResponse,
  PerClassResponse,
  PredictResponse,
  SampleResponse,
  SamplesInfo,
  SummaryMetrics,
} from '../types'

export const fetchHealth = () =>
  api.get<HealthResponse>('/health').then((r) => r.data)

export const fetchSummary = () =>
  api.get<SummaryMetrics>('/metrics/summary').then((r) => r.data)

export const fetchConfusionMatrix = () =>
  api.get<ConfusionMatrixResponse>('/metrics/confusion-matrix').then((r) => r.data)

export const fetchPerClass = () =>
  api.get<PerClassResponse>('/metrics/per-class').then((r) => r.data)

export const fetchSamplesInfo = () =>
  api.get<SamplesInfo>('/samples').then((r) => r.data)

export const fetchSample = (idx: number) =>
  api.get<SampleResponse>(`/samples/${idx}`).then((r) => r.data)

export const fetchRandomSample = () =>
  api.get<SampleResponse>('/samples/random').then((r) => r.data)

export const predictSignal = (signal: number[][]) =>
  api.post<PredictResponse>('/predict', { signal }).then((r) => r.data)

export const predictUpload = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api
    .post<PredictResponse>('/predict/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((r) => r.data)
}
