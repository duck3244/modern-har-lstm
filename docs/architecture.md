# Architecture — Modern HAR LSTM

UCI HAR 데이터셋의 스마트폰 관성 신호(128 timesteps × 9 channels)를 입력으로 6가지 인간 활동을 분류하는 시스템의 아키텍처를 설명한다. 학습 파이프라인(Python/TensorFlow)과 대시보드(FastAPI 서빙 + Vue 3 SPA)로 구성된다.

## 1. 시스템 개요

```
┌────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                          │
│                                                                    │
│   UCI HAR ZIP ──▶ DataLoader ──▶ DataPreprocessor ──▶ HARModel     │
│                                  (subject-aware split)  (LSTM)     │
│                                                           │        │
│                   ModelEvaluator / ErrorAnalyzer / Visualizer      │
│                                                           ▼        │
│                                     best_model.keras / results/    │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (load_model at startup)
┌────────────────────────────────────────────────────────────────────┐
│                       Serving Layer (FastAPI)                      │
│                                                                    │
│   lifespan: load model + X_test/y_test, pre-compute metrics        │
│                                                                    │
│   /api/health          /api/metrics/summary                        │
│   /api/samples/…       /api/metrics/confusion-matrix               │
│   /api/predict         /api/metrics/per-class                      │
│   /api/predict/upload                                              │
└────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │  HTTP (Vite dev-proxy on /api)
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                   Frontend SPA (Vue 3 + Pinia)                     │
│                                                                    │
│   Router ──▶ DashboardView  ──▶ stores/metrics  ──▶ api/endpoints  │
│           └▶ PredictView    ──▶ stores/samples  ──▶ api/endpoints  │
│                                                                    │
│   Components (ECharts):                                            │
│     SummaryCards · ConfusionMatrixHeatmap · PerClassBarChart       │
│     SignalChart · PredictionResult                                 │
└────────────────────────────────────────────────────────────────────┘
```

## 2. 저장소 구성

```
modern-har-lstm/
├── backend/
│   ├── config.py              # Config dataclass + ConfigPresets
│   ├── data_loader.py         # DataLoader, DataPreprocessor
│   ├── model.py               # HARModel, ModelBuilder
│   ├── evaluator.py           # ModelEvaluator, ErrorAnalyzer
│   ├── visualizer.py          # Visualizer
│   ├── main.py                # CLI 학습/평가 엔트리포인트
│   ├── best_model.keras       # 학습된 가중치 (서빙 대상)
│   └── api/
│       ├── app.py             # FastAPI 앱, lifespan, CORS, 라우터 등록
│       ├── deps.py            # AppState, 모델 경로/환경 설정
│       ├── schemas.py         # Pydantic I/O 스키마
│       └── routes/
│           ├── health.py      # GET /api/health
│           ├── metrics.py     # GET /api/metrics/{summary,confusion-matrix,per-class}
│           ├── samples.py     # GET /api/samples, /samples/{idx}, /samples/random
│           └── predict.py     # POST /api/predict, /predict/upload
├── frontend/
│   └── src/
│       ├── main.ts            # 앱 부트스트랩 (Pinia, Router, vue-echarts)
│       ├── App.vue            # 셸 레이아웃 + 내비게이션
│       ├── router/index.ts    # /dashboard · /predict
│       ├── api/
│       │   ├── client.ts      # axios 인스턴스 (baseURL=/api)
│       │   └── endpoints.ts   # fetch*/predict* 래퍼
│       ├── stores/            # Pinia: metrics, samples
│       ├── views/             # DashboardView, PredictView
│       ├── components/        # 5개 차트/카드 컴포넌트
│       └── types.ts           # 백엔드 스키마와 1:1 TS 타입
├── data/UCI HAR Dataset/      # 원본 데이터셋 (자동 다운로드)
├── results/ · visualizations/ # 평가 리포트 및 그림 산출물
└── sample_inputs/             # 예측 업로드 테스트용 샘플
```

## 3. 런타임 / 기술 스택

| Layer      | Stack                                                                 |
|------------|-----------------------------------------------------------------------|
| Training   | Python 3.10, TensorFlow 2.10 / Keras 2.10, NumPy, scikit-learn        |
| Viz        | matplotlib, seaborn (학습 시 PNG 산출)                                  |
| API        | FastAPI, Uvicorn, Pydantic v2, python-multipart                       |
| Frontend   | Vue 3 `<script setup>`, TypeScript, Vite 6, Pinia, Vue Router 4      |
| Charts     | ECharts 5 (`vue-echarts`)                                             |
| Styling    | Tailwind CSS 3, PostCSS, Autoprefixer                                 |
| HTTP       | axios (프록시: Vite dev 서버 `/api` → `127.0.0.1:8000`)                |

## 4. 백엔드 설계

### 4.1 설정 (`backend/config.py`)
- `Config` dataclass가 데이터/모델/학습 하이퍼파라미터를 중앙화한다 (`n_timesteps=128`, `n_features=9`, `n_classes=6`, `lstm_units=32`, `dropout_rate=0.3`, `l2_reg=0.001`, `epochs=100`, `batch_size=64`, `learning_rate=0.001`).
- `model_save_path`는 CWD와 무관하게 `backend/best_model.keras`를 절대경로로 가리킨다.
- `ConfigPresets`가 `quick` / `high_performance` / `lightweight` 프리셋을 팩토리로 제공한다.

### 4.2 데이터 로딩 (`backend/data_loader.py`)
- `DataLoader.download_dataset()`이 UCI HAR ZIP을 내려받아 `data/`에 해제한다.
- `load_signals(subset)`이 9개 관성 신호 파일을 `(n_samples, 128, 9)` 텐서로 스택한다.
- `load_labels(subset)`은 1-based 라벨을 0-based로 변환한다.
- `load_subjects(subset)`은 `GroupShuffleSplit` 기반 subject-aware split에 사용할 피험자 ID를 반환한다.
- `DataPreprocessor`가 3종 정규화(standard / minmax / robust), `subject_aware_split`, `create_sliding_windows` 를 제공한다.

### 4.3 모델 (`backend/model.py`)
- `HARModel`은 기본 아키텍처 `LSTM(32) → LSTM(32) → Dense(32, relu) → Dropout → Dense(6, softmax)`를 생성·컴파일·학습·평가·저장/로드까지 캡슐화한다. `recurrent_dropout=0`으로 두어 cuDNN 커널 최적화 경로를 유지한다.
- `get_callbacks()`: `EarlyStopping(val_loss, patience=15)`, `ReduceLROnPlateau(val_loss, factor=0.5, patience=10)`, `ModelCheckpoint(save_best_only=True)`.
- `ModelBuilder`는 `simple` / `deep` / `bidirectional` LSTM 변형을 팩토리 정적 메서드로 노출한다.

### 4.4 평가·시각화
- `ModelEvaluator`: accuracy, confusion matrix, per-class precision/recall/F1, macro·weighted 평균, classification report, confidence/ROC 분석, 성능 리포트 및 JSON/CSV 저장.
- `ErrorAnalyzer`: 오분류 패턴, 어려운 샘플(고신뢰도 오류), 클래스별 오류율.
- `Visualizer`: 훈련 곡선, 혼동 행렬, 분류 리포트 히트맵, 클래스 분포, 신호 통계, 예측 신뢰도, 샘플 신호 파형 등 PNG 출력.

### 4.5 CLI 엔트리포인트 (`backend/main.py`)
- `argparse`로 프리셋/에포크/배치/모델 타입/시각화 저장/스킵 플래그 등을 받는다.
- `set_global_seed(seed)`로 Python/NumPy/TF 시드를 고정해 재현성을 확보한다.
- 실행 흐름: ① 설정 구성 → ② 데이터 다운로드/로드 → ③ subject-aware train/val split → ④ 모델 생성·학습(또는 체크포인트 로드) → ⑤ 평가 및 오류 분석 → ⑥ 시각화 → ⑦ 결과 저장.

### 4.6 서빙 API (`backend/api/`)
- `api/app.py`
  - `setup_environment()`을 **tensorflow import 이전**에 호출해 TF 로그 억제와 `CUDA_VISIBLE_DEVICES=-1`로 CPU-only 추론을 강제한다.
  - `lifespan` 훅에서 ① 모델 로드(`best_model.keras`→`.h5` fallback), ② 테스트셋 로드, ③ 1-batch 워밍업, ④ 전체 테스트셋 예측으로 메트릭 캐시(`AppState.metrics`) 구축.
  - CORS는 Vite dev 오리진(`http://localhost:5173`, `127.0.0.1:5173`)에 한해 허용.
  - 라우터 네임스페이스: `/api/health`, `/api/metrics`, `/api/samples`, `/api/predict`.
- `api/deps.py`
  - `AppState` 데이터클래스 하나를 모듈 싱글턴(`state`)으로 유지 — 라우트들은 이 객체를 통해 모델·라벨·테스트셋·메트릭 캐시에 접근한다.
  - `resolve_model_path()`이 `best_model.keras` → `best_model.h5` 순으로 탐색한다.
- `api/schemas.py`
  - `PredictRequest.signal`을 `conlist(len=128, inner len=9)`로 런타임 검증 — 잘못된 shape 입력은 Pydantic 단계에서 422로 거부된다.
- `api/routes/predict.py`
  - JSON 본문(`POST /api/predict`)과 멀티파트 업로드(`/predict/upload`)를 모두 지원.
  - 업로드는 확장자에 따라 JSON / CSV / TXT 파서를 선택하고, 확장자가 불분명하면 내용을 보고 자동 판별한다. CSV 헤더 행, 주석(`#`)으로 시작하는 라인은 스킵.
  - `_predict_array`가 shape/NaN 검증 후 `model.predict(arr[None, ...])`로 추론하여 top-1 클래스·신뢰도·클래스별 확률 맵을 반환한다.
- `api/routes/metrics.py` / `samples.py` / `health.py`는 `state`에 미리 적재된 값을 읽어 반환하는 얇은 래퍼이다. 메트릭은 startup 1회만 계산되어 요청마다 재계산 비용이 없다.

## 5. 프론트엔드 설계

### 5.1 앱 부트스트랩
- `main.ts`: Pinia, Vue Router, 전역 `<v-chart>` 컴포넌트 등록.
- `App.vue`: 헤더 내비게이션과 `<RouterView>`로 구성된 셸.
- `router/index.ts`: `/` → `/dashboard` 리다이렉트, `/dashboard`·`/predict`를 lazy 로드.

### 5.2 HTTP 계층
- `api/client.ts`: axios 인스턴스 (`baseURL: '/api'`, `timeout: 30s`). 개발 시 Vite 프록시가 `/api`를 백엔드로 포워드한다.
- `api/endpoints.ts`: 서버의 8개 엔드포인트를 타입 안전한 단일 함수로 래핑.

### 5.3 상태 관리 (Pinia)
- `stores/metrics.ts`: `summary` / `confusion` / `perClass`를 `Promise.all`로 병렬 로드하며, 이미 로드된 경우 재호출을 생략 (`force=true`로 강제 새로고침 가능).
- `stores/samples.ts`: 테스트셋 총량과 라벨 목록을 1회 캐싱.

### 5.4 뷰·컴포넌트
- `DashboardView.vue`는 `useMetricsStore`를 `onMounted`에서 구독해 `SummaryCards`·`ConfusionMatrixHeatmap`·`PerClassBarChart`를 렌더한다.
- `PredictView.vue`는 세 가지 입력 경로를 제공한다: ① 인덱스로 테스트 샘플 선택 → `/samples/{idx}` → `/predict`, ② 랜덤 샘플, ③ 파일 업로드(`/predict/upload`). axios 에러에서 FastAPI의 `detail` 문자열을 꺼내 사용자에게 표시한다.
- 차트 컴포넌트 (ECharts) : 혼동 행렬은 normalized/counts 토글, per-class는 precision/recall/F1 그룹 막대, 신호는 9채널 3×3 small-multiples 라인, 예측 결과는 클래스별 확률 수평 막대.

### 5.5 타입 안전성
- `frontend/src/types.ts`가 백엔드 Pydantic 응답과 1:1 매칭되는 TypeScript 인터페이스를 정의한다. 서버 스키마 변경 시 이 파일만 수정하면 컴파일러가 호출부의 깨진 사용처를 지적한다.

## 6. 데이터 플로우

### 6.1 학습
1. `backend/main.py` 실행 → `Config` / `ConfigPresets` 구성 및 시드 고정.
2. `DataLoader.download_dataset()` → `load_data()` → `(X_train, y_train, X_test, y_test)` + `subjects_train`.
3. `DataPreprocessor.subject_aware_split(...)` 으로 train/val을 피험자 단위로 분리(유출 방지).
4. `HARModel.train()` — Adam + sparse categorical crossentropy; `ModelCheckpoint(save_best_only=True)`가 `backend/best_model.keras`를 갱신.
5. `ModelEvaluator.evaluate_model(...)` + `ErrorAnalyzer` 리포트를 stdout/`results/`에 기록, `Visualizer`가 PNG를 `visualizations/`에 저장.

### 6.2 서빙·추론
1. `uvicorn backend.api.app:app` 기동 → `lifespan`이 모델·테스트셋·메트릭 캐시 준비.
2. 대시보드: `DashboardView`가 `/api/metrics/{summary,confusion-matrix,per-class}`를 병렬 호출 → 차트 렌더.
3. 예측: `PredictView`가 샘플 조회/랜덤/업로드 경로로 신호를 확보 → `POST /api/predict`(JSON) 또는 `/api/predict/upload`(multipart) → `PredictionResult` · `SignalChart` 렌더.

## 7. 주요 설계 결정

- **Subject-aware split**: 피험자 단위 `GroupShuffleSplit`로 windows leakage를 차단한다. UCI HAR에서 동일 피험자가 train/val에 동시 등장하면 성능이 낙관적으로 왜곡되기 때문.
- **Startup에 메트릭 선계산**: confusion matrix 등은 요청-응답 경로에서 계산하지 않고 `lifespan`에서 1회 계산 후 캐시. 라우트는 O(1) 조회만 수행.
- **경로 독립 모델 저장**: `Config.model_save_path`를 `backend/` 기준 절대경로로 두어 학습과 서빙이 동일한 아티팩트를 공유한다.
- **TF 환경 선행 초기화**: `backend/api/app.py` 최상단에서 `setup_environment()`를 먼저 호출 — `TF_CPP_MIN_LOG_LEVEL`과 `CUDA_VISIBLE_DEVICES`는 TF import 이전에 세팅되어야 적용된다.
- **파일 업로드 포맷 관용성**: 확장자/내용 모두로 JSON·CSV·TXT를 판별하고, 숫자가 아닌 첫 행을 헤더로 간주해 스킵.
- **LSTM cuDNN 경로 유지**: `recurrent_dropout=0`을 유지하고 dropout은 입력 드롭아웃 형태로만 걸어 GPU cuDNN 커널 최적화를 활용.
- **SPA ↔ API 타입 동기화**: `types.ts`가 백엔드 Pydantic과 이름·형태를 맞춰 둬 경계에서 타입 불일치를 컴파일 타임에 검출.

## 8. 실행 방법 요약

```bash
# 학습/평가 (CLI)
python -m backend.main --config default --model_type standard --save_visualizations

# API 서버
uvicorn backend.api.app:app --port 8000 --reload

# 프론트엔드 (개발)
cd frontend && npm install && npm run dev   # http://localhost:5173
```
