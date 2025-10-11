# 현대적인 LSTM을 이용한 인간 활동 인식 시스템

TensorFlow 2.0과 LSTM 신경망을 사용하여 스마트폰 센서 데이터 기반으로 인간 활동을 분류하는 모듈형 현대적 구현체입니다.

## 🎯 개요

이 프로젝트는 딥러닝을 사용하여 6가지 인간 활동을 분류합니다:
- **걷기 (WALKING)**
- **계단 올라가기 (WALKING_UPSTAIRS)** 
- **계단 내려가기 (WALKING_DOWNSTAIRS)**
- **앉기 (SITTING)**
- **서기 (STANDING)**
- **눕기 (LAYING)**

스마트폰의 가속도계와 자이로스코프 데이터를 사용하여 91% 이상의 정확도로 예측합니다.

## 🚀 주요 기능

- **현대적인 TensorFlow 2.0**: Keras API를 사용한 깔끔하고 읽기 쉬운 코드
- **모듈형 아키텍처**: 데이터 로딩, 모델 구축, 시각화, 평가를 위한 독립적인 모듈
- **다양한 모델 타입**: Standard LSTM, Simple LSTM, Deep LSTM, Bidirectional LSTM
- **종합적인 분석**: 오류 분석과 시각화를 포함한 상세 평가
- **설정 프리셋**: 빠른 테스트, 고성능, 경량화 설정
- **명령줄 인터페이스**: 다양한 옵션을 가진 사용하기 쉬운 CLI
- **자동 데이터 다운로드**: UCI HAR 데이터셋 자동 다운로드
- **풍부한 시각화**: 훈련 그래프, 혼동 행렬, 신호 분석 등

## 📊 데이터셋

**UCI Human Activity Recognition Using Smartphones** 데이터셋을 사용합니다:
- **30명의 피험자**가 스마트폰을 착용하고 활동 수행
- **9개의 센서 신호** (3축 가속도계 + 3축 자이로스코프 + 3축 총 가속도)
- 샘플당 **128 타임스텝** (50Hz에서 2.56초)
- **7,352개의 훈련 샘플**과 **2,947개의 테스트 샘플**

*다음 작업들이 자동으로 수행됩니다:* 
- Python 버전 호환성 확인
- 가상 환경 생성
- 모든 의존성 설치
- 필요한 디렉토리 생성
- GPU 가용성 확인

### 다양한 모델 아키텍처
```bash

# 커스텀 훈련
python3 main.py --epochs 50 --batch_size 128 --learning_rate 0.001 --save_visualizations

# 양방향 LSTM
python3 main.py --model_type bidirectional

# 단순 LSTM
python3 main.py --model_type simple

# 깊은 LSTM
python3 main.py --model_type deep
```

## 📁 프로젝트 구조

```
modern-har-lstm/
├── config.py              # 설정 관리
├── data_loader.py          # 데이터 로딩 및 전처리
├── model.py               # LSTM 모델 정의
├── visualizer.py          # 시각화 유틸리티
├── evaluator.py           # 모델 평가 및 분석
├── main.py               # 메인 실행 스크립트
├── requirements.txt      # Python 의존성
├── setup.sh             # 설정 및 실행 스크립트
├── README.md           # 이 파일
├── data/               # 데이터셋 디렉토리 (자동 생성)
├── results/            # 결과 및 보고서 (자동 생성)
└── visualizations/     # 저장된 그래프 (자동 생성)
```

## ⚙️ 설정

유연한 설정 시스템과 프리셋을 제공합니다:

### 설정 프리셋

- **Default (기본)**: 일반적인 사용을 위한 균형잡힌 설정
- **Quick Test (빠른 테스트)**: 빠른 테스트를 위한 축소된 파라미터 (10 에포크)
- **High Performance (고성능)**: 최고 결과를 위한 확장된 훈련 (200 에포크)
- **Lightweight (경량화)**: 배포를 위한 최소 모델

### 커스텀 설정

```python
python3 main.py --help
```

사용 가능한 옵션들:
- `--config`: 설정 프리셋 (default, quick, high_performance, lightweight)
- `--epochs`: 훈련 에포크 수
- `--batch_size`: 훈련 배치 크기
- `--learning_rate`: 학습률
- `--lstm_units`: LSTM 유닛 수
- `--model_type`: 모델 아키텍처 (standard, simple, deep, bidirectional)
- `--save_visualizations`: 그래프를 파일로 저장
- `--skip_training`: 훈련 대신 기존 모델 로드

## 🏗 모델 아키텍처

### Standard LSTM (기본)
- 2개의 스택된 LSTM 레이어 (각각 32 유닛)
- ReLU 활성화 함수를 가진 Dense 레이어
- 정규화를 위한 Dropout
- L2 정규화

### Simple LSTM
- 단일 LSTM 레이어
- 직접 출력 분류
- 빠른 훈련, 약간 낮은 정확도

### Deep LSTM
- 3개의 스택된 LSTM 레이어
- 더 복잡한 패턴 인식
- 더 긴 훈련 시간

### Bidirectional LSTM
- 양방향으로 시퀀스 처리
- 더 나은 컨텍스트 이해
- 높은 계산 비용

## 📈 결과

예상 성능 지표:
- **정확도**: ~91-93%
- **정밀도**: ~92%
- **재현율**: ~92%
- **F1-점수**: ~92%

### 샘플 출력
```
🎯 전체 성능:
  정확도: 0.9165 (91.65%)
  매크로 평균 정밀도: 0.9176
  매크로 평균 재현율: 0.9165
  매크로 평균 F1-점수: 0.9164

📊 클래스별 성능:
  걷기:
    정밀도: 0.940
    재현율: 0.939
    F1-점수: 0.940
  앉기:
    정밀도: 0.899
    재현율: 0.831
    F1-점수: 0.864
  ...
```

## 📊 시각화

프로젝트는 종합적인 시각화를 생성합니다:

1. **훈련 기록**: 손실 및 정확도 곡선
2. **혼동 행렬**: 상세한 분류 성능
3. **분류 보고서**: 정밀도, 재현율, F1-점수 히트맵
4. **신호 분석**: 센서 데이터의 통계적 특성
5. **예측 신뢰도**: 신뢰도 분포 분석
6. **클래스 분포**: 데이터셋 균형 시각화
7. **샘플 신호**: 개별 신호 파형

## 📋 평가 기능

### 종합적인 지표
- 정확도, 정밀도, 재현율, F1-점수
- 클래스별 성능 분석
- 상세한 분해를 포함한 혼동 행렬
- 다중 클래스 분류를 위한 ROC AUC 점수

### 오류 분석
- 오분류 패턴 식별
- 어려운 샘플 분석 (고신뢰도 오류)
- 클래스별 오류율
- 올바른 vs 잘못된 예측의 신뢰도 분석

### 자동 보고서
- 텍스트 파일로 저장되는 성능 보고서
- JSON 형식의 상세 결과
- CSV 파일로 된 혼동 행렬
- 실험 추적을 위한 타임스탬프 결과

## 🔧 고급 사용법

### 기존 모델 로드 및 평가
```bash
python3 main.py --skip_training --model_path your_model.h5
```

### 커스텀 모델 아키텍처
```python
from model import ModelBuilder
from config import Config

config = Config()
custom_model = ModelBuilder.build_deep_lstm(config, num_layers=4)
```

### 데이터 전처리
```python
from data_loader import DataPreprocessor

preprocessor = DataPreprocessor()
X_normalized, params = preprocessor.normalize_data(X_train, method='standard')
```

### 모델 비교
```python
from evaluator import ModelEvaluator

evaluator = ModelEvaluator(config)
comparison_df = evaluator.compare_models(results_list, model_names)
```

## 🎨 커스터마이징

### 새로운 모델 타입 추가
1. `model.py`에 새로운 모델 빌더 생성
2. `ModelBuilder` 클래스에 추가
3. `main.py`에서 CLI 옵션 업데이트

### 커스텀 시각화
1. `Visualizer` 클래스에 메서드 추가
2. 메인 실행 플로우에서 호출
3. 필요에 따라 저장 경로 설정

### 새로운 평가 지표
1. `ModelEvaluator` 클래스 확장
2. 평가 파이프라인에 추가
3. 보고서에 포함

---

**즐거운 활동 인식! 🏃‍♀️🚶‍♂️**