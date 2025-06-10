import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.font_manager as fm

# 한국어 폰트 (윈도우 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 로드
@st.cache_resource
def load_model():
    with open("xgb_fire_model_ver2.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.markdown("""
<div class='main-title-card'>
    <h1>🔥 산불 예측 대시보드</h1>
    <div class='desc'>기상 및 산불 위험 데이터를 기반으로 산불 발생 가능성을 예측합니다.</div>
    <div class='guide'>예측할 데이터를 CSV로 업로드하거나 수도로 입력해보세요.</div>
</div>
""", unsafe_allow_html=True)

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='cp949')

    column_mapping = {
        '평균기온(℃)': 'TA_AVG',
        '평균습도(%rh)': 'HM_AVG',
        '실효습도': 'effective_humidity',
        '풍속': 'WS_AVG',
        '기온': 'TA_AVG',
        '습도': 'HM_AVG',
        '풍속(m/s)': 'WS_AVG',
        'wind_avg': 'WS_AVG',
        'temp_avg': 'TA_AVG',
        'humidity': 'HM_AVG'
    }
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

    if 'WS_AVG' not in df.columns:
        st.error("❌ 필수 컬럼 'WS_AVG'가 없습니다.")
        st.stop()

    df["wind_speed"] = df["WS_AVG"]

    st.markdown("### 📌 업로드한 데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 📈 주요 통계")
    st.dataframe(df.describe().T)

    if st.button("🚀 산불 발생 예측하기"):
        try:
            model_features = ['WS_AVG', 'HM_AVG', 'effective_humidity', 'wind_speed', 'TA_AVG']
            X = df[model_features].fillna(df[model_features].mean())
            preds_proba = model.predict_proba(X)[:, 1]
            preds = model.predict(X)

            df["산불발생확률(%)"] = np.round(preds_proba * 100, 2)

            st.success("✅ 예측이 완료되었습니다!")
            st.dataframe(df[[*model_features, "산불발생확률(%)"]], use_container_width=True)

            if '산불발생여부' in df.columns:
                st.markdown("### 📏 예측 성능 평가")
                y_true = df['산불발생여부']
                y_pred = preds

                acc = accuracy_score(y_true, y_pred)
                st.metric("정확도 (Accuracy)", f"{acc * 100:.2f}%")

                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                total = tn + fp + fn + tp
                correct = tn + tp
                wrong = fp + fn

                st.write(f"✅ 총 {total}개 중 {correct}개 맞음 (정답률: {acc * 100:.2f}%)")
                st.write(f"❌ 틀린 개수: {wrong}개 (False Positive: {fp}, False Negative: {fn})")

                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

                st.markdown("### 🔍 혼동 행렬 (Confusion Matrix)")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
                ax_cm.set_xlabel("예측 값")
                ax_cm.set_ylabel("실제 값")
                st.pyplot(fig_cm)
            else:
                st.info("📌 참고: 예측 평가 지표는 '산불발생여부' 컬럼이 있을 때만 확인가능해요.")

            # 변수 중요도 시각화 (한글 표기)
            st.markdown("### 📌 모델 변수 중요도 (XGBoost 기준)")
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            korean_labels = {
                'HM_AVG': '평균 습도',
                'TA_AVG': '평균 기온',
                'effective_humidity': '실효습도',
                'WS_AVG': '평균 풍속',
                'wind_speed': '풍속(중복)'
            }
            importance_df = pd.DataFrame([
                {
                    '변수': korean_labels.get(k, k),
                    '중요도': round(v, 1)
                } for k, v in importance_dict.items() if k in korean_labels
            ]).sort_values(by='중요도', ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(importance_df['변수'], importance_df['중요도'], color='skyblue')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va='center')
            ax.set_title("변수 중요도 (XGBoost - Gain 기준)")
            ax.set_xlabel("중요도 점수")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠ 예측 중 오류 발생: {e}")

else:
    st.markdown("### 📄 사용 방법")
    st.markdown("""
    1. 기상 데이터를 포함한 CSV 파일을 업로드합니다.  
    2. 파일은 `평균기온`, `습도`, `실효습도`, `풍속` 등의 컬럼을 포함해야 합니다.  
    3. 업로드 후 '산불 발생 예측하기' 버튼을 누르면 예측 결과가 표시됩니다.  
    """)

# 수동 입력
st.markdown("---")
st.markdown("## 📝 수동 입력으로 산불 예측하기")

ta_avg = st.number_input("평균 기온 (℃)", min_value=-30.0, max_value=50.0, value=20.0)
hm_avg = st.number_input("평균 습도 (%)", min_value=0.0, max_value=100.0, value=50.0)
effective_humidity = st.number_input("실효습도", min_value=0.0, max_value=100.0, value=30.0)
ws_avg = st.number_input("평균 풍속 (m/s)", min_value=0.0, max_value=50.0, value=2.0)

if st.button("🔥 입력값으로 산불 예측하기"):
    input_data = pd.DataFrame([{
        'WS_AVG': ws_avg,
        'HM_AVG': hm_avg,
        'effective_humidity': effective_humidity,
        'wind_speed': ws_avg,
        'TA_AVG': ta_avg
    }])
    try:
        prob = model.predict_proba(input_data)[0][1]
        prob_percent = prob * 100

        if prob_percent >= 80:
            risk_level = "🚨 매우 높음"
        elif prob_percent >= 60:
            risk_level = "🔶 높음"
        elif prob_percent >= 40:
            risk_level = "🟡 보통"
        elif prob_percent >= 20:
            risk_level = "🟢 낮음"
        else:
            risk_level = "🔵 매우 낮음"

        st.success(f"🌲 산불 발생 확률: **{prob_percent:.2f}%**")
        st.info(f"📊 위험 등급: {risk_level}")

        explanation = f"""
        ### 🔍 판단 근거
        - 평균 습도(HM_AVG): {hm_avg}% → 낮을수록 산불 가능성 ↑
        - 평균 기온(TA_AVG): {ta_avg}℃ → 높을수록 산불 가능성 ↑
        - 실효습도: {effective_humidity}% → 낮을수록 건조 → 산불 위험 ↑
        - 평균 풍속: {ws_avg}m/s → 강할수록 확산 위험 존재
        """
        st.markdown(explanation)

    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {e}")