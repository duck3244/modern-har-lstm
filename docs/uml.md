# UML — Modern HAR LSTM

Mermaid 기반 다이어그램 모음. GitHub·PyCharm·VS Code의 Mermaid 뷰어에서 바로 렌더된다. 구조는 `architecture.md` 와 짝을 이룬다.

## 1. Component Diagram — 시스템 구성

```mermaid
flowchart LR
  subgraph Training["Training Pipeline (python -m backend.main)"]
    CLI["main.py<br/>(argparse + seed + orchestrator)"]
    CFG["Config / ConfigPresets"]
    DL["DataLoader"]
    DP["DataPreprocessor"]
    MB["ModelBuilder"]
    HM["HARModel"]
    EV["ModelEvaluator"]
    ER["ErrorAnalyzer"]
    VZ["Visualizer"]
  end

  DS[("UCI HAR Dataset<br/>(ZIP/TXT)")]
  MODEL[("best_model.keras")]
  RESULTS[("results/ · visualizations/")]

  CLI --> CFG
  CLI --> DL
  CLI --> DP
  CLI --> HM
  CLI --> EV
  CLI --> ER
  CLI --> VZ
  DL --> DS
  HM --> MB
  HM -->|ModelCheckpoint| MODEL
  EV --> RESULTS
  VZ --> RESULTS

  subgraph API["FastAPI Serving (backend/api)"]
    APP["app.py<br/>(lifespan · CORS · routers)"]
    DEPS["deps.py<br/>(AppState singleton)"]
    SCH["schemas.py (Pydantic)"]
    RH["routes/health"]
    RM["routes/metrics"]
    RS["routes/samples"]
    RP["routes/predict"]
  end

  APP --> DEPS
  APP --> RH
  APP --> RM
  APP --> RS
  APP --> RP
  RH --> DEPS
  RM --> DEPS
  RS --> DEPS
  RP --> DEPS
  RP --> SCH
  APP -->|load_model| MODEL
  APP -->|load X_test/y_test| DS

  subgraph FE["Frontend SPA (Vue 3 + Pinia)"]
    MAIN["main.ts"]
    APPVUE["App.vue"]
    RTR["router/index.ts"]
    DV["views/DashboardView"]
    PV["views/PredictView"]
    SM["stores/metrics"]
    SS["stores/samples"]
    EP["api/endpoints.ts"]
    AX["api/client.ts (axios)"]
    CMP["components/*"]
  end

  MAIN --> APPVUE
  APPVUE --> RTR
  RTR --> DV
  RTR --> PV
  DV --> SM
  DV --> CMP
  PV --> SS
  PV --> EP
  PV --> CMP
  SM --> EP
  SS --> EP
  EP --> AX
  AX -.HTTP /api.-> APP
```

## 2. Class Diagram — 백엔드 핵심 클래스

```mermaid
classDiagram
  class Config {
    +int n_timesteps = 128
    +int n_features = 9
    +int n_classes = 6
    +int lstm_units = 32
    +float dropout_rate = 0.3
    +float l2_reg = 0.001
    +int batch_size = 64
    +int epochs = 100
    +float learning_rate = 0.001
    +float validation_split = 0.2
    +Path data_dir
    +Path dataset_dir
    +str model_save_path
    +str results_save_path
    +List~str~ labels
    +List~str~ signal_types
    +create_directories() void
    +get_model_params() dict
    +update_config(**kwargs) void
  }

  class ConfigPresets {
    <<static>>
    +get_quick_test_config() Config
    +get_high_performance_config() Config
    +get_lightweight_config() Config
  }

  class DataLoader {
    -Config config
    +download_dataset() void
    +load_signals(subset) ndarray
    +load_labels(subset) ndarray
    +load_subjects(subset) ndarray
    +load_data() tuple
    +get_data_info() dict
  }

  class DataPreprocessor {
    <<static>>
    +normalize_data(X, method) tuple
    +apply_normalization(X, params) ndarray
    +subject_aware_split(X, y, subjects, val_fraction, seed) tuple
    +create_sliding_windows(X, y, window_size, step_size) tuple
  }

  class HARModel {
    -Config config
    -Model model
    -History history
    -bool is_compiled
    +build_model() Model
    +compile_model() void
    +get_callbacks() list
    +train(X_train, y_train, X_val, y_val) void
    +evaluate(X_test, y_test) dict
    +predict(X) tuple
    +save_model(filepath) void
    +load_model(filepath) void
    +get_model_summary() str
  }

  class ModelBuilder {
    <<static>>
    +build_simple_lstm(config) Model
    +build_deep_lstm(config, num_layers) Model
    +build_bidirectional_lstm(config) Model
  }

  class ModelEvaluator {
    -Config config
    +evaluate_model(y_true, y_pred, y_pred_proba) dict
    +generate_performance_report(results, model_params) str
    +save_results(results, model_params) void
    +compare_models(results_list, names) DataFrame
  }

  class ErrorAnalyzer {
    -Config config
    +generate_error_report(X, y_true, y_pred, y_pred_proba) str
  }

  class Visualizer {
    -Config config
    +plot_training_history(history, save_path) void
    +plot_confusion_matrix(y_true, y_pred, normalize, save_path) void
    +plot_classification_report(y_true, y_pred, save_path) void
    +plot_class_distribution(y, subset_name, save_path) void
    +plot_signal_statistics(X, save_path) void
    +plot_prediction_confidence(y_true, y_pred_proba, save_path) void
    +plot_signal_data(sample, activity_label, save_path) void
    +create_comprehensive_report(results, out_dir) void
  }

  HARModel --> Config
  HARModel ..> ModelBuilder : delegates (optional)
  DataLoader --> Config
  ModelEvaluator --> Config
  ErrorAnalyzer --> Config
  Visualizer --> Config
  ConfigPresets ..> Config : factory
```

## 3. Class Diagram — API 계층

```mermaid
classDiagram
  class AppState {
    +object model
    +List~str~ labels
    +ndarray X_test
    +ndarray y_test
    +dict metrics
  }

  class PredictRequest {
    +List~List~float~~ signal  // 128 × 9 (conlist)
  }

  class PredictResponse {
    +int predicted_class
    +str predicted_label
    +float confidence
    +Dict~str,float~ probabilities
  }

  class SampleResponse {
    +int index
    +List~List~float~~ signal
    +int true_class
    +str true_label
  }

  class SamplesInfoResponse {
    +int total
    +List~str~ labels
  }

  class SummaryMetrics {
    +float accuracy
    +float macro_precision
    +float macro_recall
    +float macro_f1
    +float weighted_precision
    +float weighted_recall
    +float weighted_f1
    +int total_samples
  }

  class ConfusionMatrixResponse {
    +List~str~ labels
    +List~List~int~~ matrix
    +List~List~float~~ normalized
  }

  class PerClassRow {
    +str label
    +float precision
    +float recall
    +float f1
    +int support
  }

  class PerClassResponse {
    +List~PerClassRow~ rows
  }

  class HealthResponse {
    +str status
    +str model_path
    +int n_test_samples
    +List~str~ labels
  }

  class FastAPIApp {
    +lifespan(app) context
    +CORSMiddleware
    +include_router(health, metrics, samples, predict)
  }

  class HealthRoute {
    +GET /api/health ~HealthResponse~
  }
  class MetricsRoute {
    +GET /api/metrics/summary ~SummaryMetrics~
    +GET /api/metrics/confusion-matrix ~ConfusionMatrixResponse~
    +GET /api/metrics/per-class ~PerClassResponse~
  }
  class SamplesRoute {
    +GET /api/samples ~SamplesInfoResponse~
    +GET /api/samples/:idx ~SampleResponse~
    +GET /api/samples/random ~SampleResponse~
  }
  class PredictRoute {
    +POST /api/predict ~PredictResponse~
    +POST /api/predict/upload ~PredictResponse~
  }

  FastAPIApp --> AppState : lifespan 주입
  HealthRoute --> AppState
  MetricsRoute --> AppState
  SamplesRoute --> AppState
  PredictRoute --> AppState
  PredictRoute ..> PredictRequest
  PredictRoute ..> PredictResponse
  SamplesRoute ..> SampleResponse
  SamplesRoute ..> SamplesInfoResponse
  MetricsRoute ..> SummaryMetrics
  MetricsRoute ..> ConfusionMatrixResponse
  MetricsRoute ..> PerClassResponse
  HealthRoute ..> HealthResponse
  PerClassResponse o-- PerClassRow
```

## 4. Class Diagram — 프론트엔드 모듈

```mermaid
classDiagram
  class AxiosClient {
    +baseURL = /api
    +timeout = 30000
  }

  class Endpoints {
    <<module>>
    +fetchHealth() HealthResponse
    +fetchSummary() SummaryMetrics
    +fetchConfusionMatrix() ConfusionMatrixResponse
    +fetchPerClass() PerClassResponse
    +fetchSamplesInfo() SamplesInfo
    +fetchSample(idx) SampleResponse
    +fetchRandomSample() SampleResponse
    +predictSignal(signal) PredictResponse
    +predictUpload(file) PredictResponse
  }

  class MetricsStore {
    <<Pinia>>
    +Ref~SummaryMetrics~ summary
    +Ref~ConfusionMatrixResponse~ confusion
    +Ref~PerClassResponse~ perClass
    +Ref~bool~ loading
    +Ref~string~ error
    +load(force) Promise
  }

  class SamplesStore {
    <<Pinia>>
    +Ref~SamplesInfo~ info
    +load() Promise
  }

  class Router {
    +"/" → redirect /dashboard
    +"/dashboard" → DashboardView
    +"/predict" → PredictView
  }

  class AppShell {
    <<App.vue>>
    +RouterLink Dashboard
    +RouterLink Predict
    +RouterView
  }

  class DashboardView {
    -MetricsStore store
    +onMounted → store.load()
  }

  class PredictView {
    -SamplesStore samplesStore
    -Ref~number~ idxInput
    -Ref~SampleResponse~ sample
    -Ref~PredictResponse~ prediction
    +loadByIndex()
    +loadRandom()
    +onFileSelected(event)
  }

  class SummaryCards
  class ConfusionMatrixHeatmap
  class PerClassBarChart
  class SignalChart
  class PredictionResult

  AppShell --> Router
  Router --> DashboardView
  Router --> PredictView
  DashboardView --> MetricsStore
  DashboardView --> SummaryCards
  DashboardView --> ConfusionMatrixHeatmap
  DashboardView --> PerClassBarChart
  PredictView --> SamplesStore
  PredictView --> Endpoints
  PredictView --> SignalChart
  PredictView --> PredictionResult
  MetricsStore --> Endpoints
  SamplesStore --> Endpoints
  Endpoints --> AxiosClient
```

## 5. Sequence Diagram — 학습 실행 흐름

```mermaid
sequenceDiagram
  autonumber
  actor User
  participant CLI as main.py
  participant CFG as Config/Presets
  participant DL as DataLoader
  participant DP as DataPreprocessor
  participant M as HARModel
  participant E as ModelEvaluator
  participant V as Visualizer
  participant FS as filesystem

  User->>CLI: python -m backend.main --config default
  CLI->>CFG: setup_config(args) · create_directories()
  CLI->>DL: download_dataset()
  DL->>FS: urlretrieve + unzip → data/UCI HAR Dataset/
  CLI->>DL: load_data() → X_train, y_train, X_test, y_test
  CLI->>DL: load_subjects("train")
  CLI->>DP: subject_aware_split(X_train, y_train, subjects, 0.2)
  DP-->>CLI: X_tr, y_tr, X_val, y_val
  CLI->>M: create_model(config, model_type)
  M->>M: build_model() · compile_model()
  CLI->>M: train(X_tr, y_tr, X_val, y_val)
  M->>FS: ModelCheckpoint → backend/best_model.keras
  M-->>CLI: history
  CLI->>M: evaluate(X_test, y_test)
  M-->>CLI: {test_loss, test_accuracy, predictions, probs, true}
  CLI->>E: evaluate_model(y_true, y_pred, y_proba)
  E-->>CLI: comprehensive_results
  CLI->>V: plot_*  (confusion, per-class, signal, confidence, ...)
  V->>FS: PNGs → visualizations/
  CLI->>E: save_results(all_results, model_params)
  E->>FS: JSON / CSV / TXT → results/
```

## 6. Sequence Diagram — 대시보드 초기 로드

```mermaid
sequenceDiagram
  autonumber
  participant U as User (browser)
  participant V as DashboardView
  participant S as MetricsStore (Pinia)
  participant EP as api/endpoints
  participant API as FastAPI
  participant ST as AppState (cached)

  U->>V: navigate /dashboard
  V->>S: onMounted → load()
  par 병렬 요청
    S->>EP: fetchSummary()
    EP->>API: GET /api/metrics/summary
    API->>ST: read state.metrics.summary
    API-->>EP: SummaryMetrics
  and
    S->>EP: fetchConfusionMatrix()
    EP->>API: GET /api/metrics/confusion-matrix
    API->>ST: read state.metrics.confusion_matrix
    API-->>EP: ConfusionMatrixResponse
  and
    S->>EP: fetchPerClass()
    EP->>API: GET /api/metrics/per-class
    API->>ST: read state.metrics.per_class
    API-->>EP: PerClassResponse
  end
  EP-->>S: 응답 3종
  S-->>V: refs 업데이트 (reactive)
  V->>U: SummaryCards · ConfusionMatrixHeatmap · PerClassBarChart 렌더
```

## 7. Sequence Diagram — 예측 (인덱스 / 업로드)

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant PV as PredictView
  participant EP as api/endpoints
  participant API as FastAPI
  participant ST as AppState
  participant TF as Keras model

  Note over U,PV: (A) 테스트 샘플 인덱스로 예측
  U->>PV: 입력 idx + Predict 클릭
  PV->>EP: fetchSample(idx)
  EP->>API: GET /api/samples/{idx}
  API->>ST: X_test[idx], y_test[idx]
  API-->>EP: SampleResponse
  EP-->>PV: sample
  PV->>EP: predictSignal(sample.signal)
  EP->>API: POST /api/predict (JSON body)
  API->>API: PredictRequest 검증 (128×9 conlist)
  API->>TF: model.predict(arr[None, ...])
  TF-->>API: probabilities[6]
  API-->>EP: PredictResponse
  EP-->>PV: prediction
  PV->>U: PredictionResult + SignalChart 렌더

  Note over U,PV: (B) 파일 업로드로 예측
  U->>PV: 파일 선택 (.json/.csv/.txt)
  PV->>EP: predictUpload(file)
  EP->>API: POST /api/predict/upload (multipart)
  alt 확장자별 파서 분기
    API->>API: _parse_json / _parse_csv / _parse_txt
  else 불분명한 확장자
    API->>API: 내용 보고 자동 판별
  end
  API->>API: _predict_array (shape/NaN 검증)
  API->>TF: model.predict(arr[None, ...])
  TF-->>API: probabilities
  API-->>EP: PredictResponse
  EP-->>PV: prediction
  PV->>U: 결과 렌더 (sample 없음 → SignalChart 생략)
```

## 8. Sequence Diagram — API 기동 (lifespan)

```mermaid
sequenceDiagram
  autonumber
  participant UV as uvicorn
  participant APP as FastAPI app
  participant DEPS as deps.setup_environment
  participant CFG as Config
  participant DL as DataLoader
  participant TF as tf.keras
  participant SK as sklearn.metrics
  participant ST as AppState

  UV->>DEPS: setup_environment()   // TF import 이전에 실행
  UV->>APP: import app (FastAPI 생성)
  UV->>APP: lifespan start
  APP->>CFG: Config() + 경로 resolve
  APP->>DEPS: resolve_model_path()
  APP->>TF: load_model(best_model.keras|h5)
  APP->>DL: load_signals("test") · load_labels("test")
  APP->>TF: model.predict(X_test[:1])   // warm-up
  APP->>TF: model.predict(X_test, batch_size=256)
  APP->>SK: accuracy · confusion_matrix · prf_support(macro/weighted/per-class)
  APP->>ST: state.model / labels / X_test / y_test / metrics 설정
  APP-->>UV: 준비 완료 → 요청 수신 시작
  Note right of ST: 이후 라우트는 state만 읽어<br/>O(1) 응답
```

## 9. State Diagram — `AppState` 라이프사이클

```mermaid
stateDiagram-v2
  [*] --> Starting : process launched
  Starting --> Loading : lifespan enter
  Loading --> Ready : model + metrics cached
  Loading --> Failed : model file missing / load error
  Ready --> Ready : GET /api/* (read-only)
  Ready --> Predicting : POST /api/predict[/upload]
  Predicting --> Ready : 200 OK
  Predicting --> Ready : 4xx (shape / NaN / parse)
  Ready --> Shutdown : lifespan exit
  Failed --> Shutdown
  Shutdown --> [*]
```

## 10. Deployment Diagram — 개발 환경

```mermaid
flowchart LR
  subgraph Dev["Developer Machine"]
    direction LR
    BROWSER["Browser<br/>http://localhost:5173"]
    VITE["Vite Dev Server<br/>:5173<br/>(proxy /api → :8000)"]
    UVI["Uvicorn<br/>backend.api.app:app<br/>:8000"]
    PY["Python 3.10 venv (py310_tf)<br/>TF 2.10 · FastAPI · scikit-learn"]
    NODE["Node ≥18.20<br/>Vue 3 · Pinia · ECharts · Tailwind"]
    FS[("backend/best_model.keras<br/>data/UCI HAR Dataset")]
  end

  BROWSER -->|HTTP| VITE
  VITE -->|proxy /api| UVI
  UVI --> PY
  PY --> FS
  VITE --> NODE
```

---

**주석:**
- 도식의 필드/메서드는 현재 코드 기준의 공개 API에 한정했다. 내부 헬퍼(`_build_sample`, `_parse_json` 등)는 흐름 다이어그램에만 등장한다.
- 프론트엔드의 Vue 컴포넌트는 클래스가 아니지만 `<script setup>` 노출 심볼을 의사(class)로 표기했다.
