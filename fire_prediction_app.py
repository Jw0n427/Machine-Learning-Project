import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ----- 스타일 생략: 요청하신 CSS 스타일은 그대로 위에서 유지됨 (너무 길어 생략 가능) -----

# 상단 타이틀
st.markdown("""
<div class='main-title-card'>
    <h1>🔥 산불 예측 대시보드</h1>
    <div class='desc'>기상 및 산불 위험 데이터를 기반으로 산불 발생 가능성을 예측합니다.</div>
    <div class='guide'>예측할 데이터를 CSV로 업로드하세요.</div>
</div>
""", unsafe_allow_html=True)

# 모델 불러오기
def load_model():
    with open("xgb_fire_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# 파일 업로드
uploaded_file = st.file_uploader("", type=["csv"], help="예측할 데이터를 CSV로 업로드하세요.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='cp949')
    st.markdown("<h4 style='margin-top:2em; color:#333;'>업로드한 데이터 미리보기</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # 데이터 통계 요약
    st.markdown("<h5 style='margin-top:2em; color:#ff7043;'>데이터 요약</h5>", unsafe_allow_html=True)
    total_rows = len(df)
    total_cols = len(df.columns)
    null_rows = df.isnull().any(axis=1).sum()
    null_ratio = 100 * null_rows / total_rows if total_rows else 0
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("총 행 개수", f"{total_rows:,}")
    with col2: st.metric("총 열 개수", f"{total_cols:,}")
    with col3: st.metric("결측치 포함 행", f"{null_rows:,}")
    with col4: st.metric("결측치 비율(%)", f"{null_ratio:.1f}%")

    # 주요 컬럼 통계
    st.markdown("<h5 style='margin-top:2em; color:#ff7043;'>주요 컬럼 통계</h5>", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)

    # 분포 시각화
    st.markdown("<h5 style='margin-top:2em; color:#ff7043;'>주요 컬럼 분포</h5>", unsafe_allow_html=True)
    feature_cols = [col for col in ["TA_AVG", "HM_AVG", "WS_AVG", "effective_humidity", "wind_speed"] if col in df.columns]
    if feature_cols:
        fig, axs = plt.subplots(1, len(feature_cols), figsize=(4*len(feature_cols), 3))
        if len(feature_cols) == 1:
            axs = [axs]
        for i, col in enumerate(feature_cols):
            axs[i].hist(df[col].dropna(), bins=20, color="#ff7043", alpha=0.7)
            axs[i].set_title(col)
        st.pyplot(fig)
    else:
        st.info("주요 feature 컬럼이 없습니다.")

    # 예측
    if st.button("산불 발생 예측하기"):
        try:
            model_features = ['WS_AVG', 'HM_AVG', 'effective_humidity', 'wind_speed', 'TA_AVG']
            X = df[model_features]
            X = X.fillna(X.mean())
            preds = model.predict_proba(X)[:, 1]
            df["산불발생확률(%)"] = np.round(preds * 100, 2)
            st.success("예측 완료! 아래 결과를 확인하세요.")
            st.dataframe(df[[*model_features, "산불발생확률(%)"]], use_container_width=True)

            # 확률 분포
            st.markdown("<h5 style='margin-top:2em; color:#ff7043;'>산불발생확률 분포</h5>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.hist(df["산불발생확률(%)"], bins=20, color="#1976d2", alpha=0.7)
            ax2.set_xlabel("산불발생확률(%)")
            ax2.set_ylabel("건수")
            st.pyplot(fig2)

            # 상위/하위
            st.markdown("<h5 style='margin-top:2em; color:#ff7043;'>산불발생확률 TOP 5 / BOTTOM 5</h5>", unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            col5.write("**상위 5개**")
            col5.dataframe(df.sort_values("산불발생확률(%)", ascending=False).head(5), use_container_width=True)
            col6.write("**하위 5개**")
            col6.dataframe(df.sort_values("산불발생확률(%)", ascending=True).head(5), use_container_width=True)
        except KeyError as e:
            st.error(f"입력 데이터에 다음 컬럼이 없습니다: {e}")
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
else:
    st.markdown("""
    ### 프로젝트 소개
    - 기상 데이터 기반 산불 예측 머신러닝 모델 (XGBoost)
    - 주요 변수: 평균 기온, 습도, 풍속 등
    - 예측 결과: 산불발생확률(%) 표기
    """)

    st.markdown("### 예시 데이터")
    st.dataframe(pd.DataFrame({
        "TA_AVG": [-2.1, 3.5, 10.2],
        "HM_AVG": [35, 50, 70],
        "WS_AVG": [2.5, 1.8, 3.0],
        "effective_humidity": [28.5, 40.2, 55.1],
        "wind_speed": [7.1, 3.2, 5.5]
    }), use_container_width=True)

    st.markdown("""
    ### 사용 방법
    1. 예측할 데이터를 CSV로 준비
    2. 상단 업로드 영역에서 파일 선택
    3. '산불 발생 예측하기' 버튼 클릭 후 결과 확인
    """) 