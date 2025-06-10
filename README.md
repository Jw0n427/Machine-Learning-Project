# 산불 예측 머신러닝 프로젝트

이 프로젝트는 기상 데이터와 산불 위험 예보 데이터를 활용하여 산불 발생 가능성을 예측하는 머신러닝 모델을 개발합니다.

## 프로젝트 구조

- `fire_learning.ipynb`: 모델 학습 및 데이터 전처리
- `2022_fire_test.ipynb`, `2023_fire_test.ipynb`: 테스트 데이터 검증
- `xgb_fire_model.pkl`: 기본 XGBoost 모델
- `xgb_fire_model_smote.pkl`: SMOTE 기법이 적용된 XGBoost 모델

## 설치 방법

1. Python 3.8 이상이 설치되어 있어야 합니다.

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

1. 데이터 준비:
   - `산림청 국립산림과학원_대형산불위험예보목록정보_20250430.csv`
   - `산림청_산불상황관제시스템 산불통계데이터_20241016.csv`
   - `sanbul.csv`

2. 모델 학습:
   - Jupyter Notebook 실행:
   ```bash
   jupyter notebook
   ```
   - `fire_learning.ipynb` 파일을 열어 실행

3. 모델 테스트:
   - `2022_fire_test.ipynb` 또는 `2023_fire_test.ipynb` 실행

## 데이터 설명

- 기상 데이터: 기온(TA_AVG), 습도(HM_AVG), 풍속(WS_AVG)
- 산불 위험 예보 데이터: 실효습도, 풍속, 위험 등급
- 산불 발생 데이터: 발생 일시, 지역, 발생 건수

## 주의사항

- 기상청 API 키가 필요합니다 (fire_learning.ipynb의 authKey 변수 수정 필요)
- 데이터 파일의 인코딩은 'euc-kr' 또는 'cp949'를 사용합니다
