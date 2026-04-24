export interface HealthResponse {
  status: string
  model_path: string
  n_test_samples: number
  labels: string[]
}

export interface SummaryMetrics {
  accuracy: number
  macro_precision: number
  macro_recall: number
  macro_f1: number
  weighted_precision: number
  weighted_recall: number
  weighted_f1: number
  total_samples: number
}

export interface ConfusionMatrixResponse {
  labels: string[]
  matrix: number[][]
  normalized: number[][]
}

export interface PerClassRow {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface PerClassResponse {
  rows: PerClassRow[]
}

export interface SamplesInfo {
  total: number
  labels: string[]
}

export interface SampleResponse {
  index: number
  signal: number[][]
  true_class: number
  true_label: string
}

export interface PredictResponse {
  predicted_class: number
  predicted_label: string
  confidence: number
  probabilities: Record<string, number>
}
