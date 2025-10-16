import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. 加载模型与标准化器
# ===============================
# VotingRegressor 模型 (GradientBoosting : CatBoost : AdaBoost = 2 : 5 : 3)
model = joblib.load('voting_regressor.pkl')

# 训练时使用的标准化器
scaler = joblib.load('scaler.pkl')

# ===============================
# 2. Streamlit 页面标题
# ===============================
st.title("Tacrolimus Plasma Concentration Predictor")

# ===============================
# 3. 定义输入变量
# ===============================
continuous_columns = ['Total_daily_dose','CL_F','BUN','BMI','ALB','NE','CCR','IBIL','Dosing_time']  #分类变量
#columns_to_copy = ['CYP3A5']  # 分类变量

# 在 Streamlit 界面上创建输入框
st.sidebar.header("Please enter the patient's details")

Total_daily_dose = st.sidebar.number_input("Total daily dose (mg):", min_value=0.5, max_value=10.0, value=5.0)
CL_F = st.sidebar.number_input("CL/F (L/h):", min_value=15.0, max_value=30.0, value=22.5)
BUN = st.sidebar.number_input("BUN (mmol/L):", min_value=2.0, max_value=40.0, value=11.5)
BMI = st.sidebar.number_input("BMI (kg/m²):", min_value=15.0, max_value=40.0, value=24.5)
ALB = st.sidebar.number_input("ALB (g/L):", min_value=10.0, max_value=60.0, value=35.0)
NE = st.sidebar.number_input("NE# (10⁹/L):", min_value=0.5, max_value=25.0, value=6.5)
CCR = st.sidebar.number_input("CCR (mL/min):", min_value=15.0, max_value=350.0, value=115.0)
IBIL = st.sidebar.number_input("IBIL (µmol/L):", min_value=0.0, max_value=10.0, value=5.0)
Dosing_time = st.sidebar.number_input("Dosing time (day):", min_value=0.0, max_value=500.0, value=200.0)

# 汇总输入
input_data = np.array([[Total_daily_dose, CL_F, BUN, BMI, ALB, NE, CCR, IBIL,Dosing_time]])

# 转换为 DataFrame，便于后续标准化与 SHAP 解释
input_df = pd.DataFrame(input_data, columns=continuous_columns)

# ===============================
# 4. 标准化输入
# ===============================
input_scaled = scaler.transform(input_df)

# ===============================
# 5. 模型预测
# ===============================
if st.button("Predict Tacrolimus Plasma Concentration"):
    # 预测连续值
    predicted_value = model.predict(input_scaled)[0]

    # 计算 ±20% 区间
    lower_bound = predicted_value * 0.8
    upper_bound = predicted_value * 1.2

    # 输出预测结果
    st.subheader("🧪 Predicted Result")
    st.write(f"**Tacrolimus Plasma Concentration = {predicted_value:.2f} ± 20% ng/mL**")
    st.write(f"Estimated range: {lower_bound:.2f} – {upper_bound:.2f} ng/mL")

    # ===============================
    # 6. SHAP 力图解释
    # ===============================
    st.subheader("🔍 SHAP Force Plot Explanation")

    # 使用 Explainer 解释模型（适用于任意回归模型）
    # 注意：使用训练数据中的样本子集可以显著加快速度
    df_train = pd.read_csv('train.csv', encoding='utf-8')  # 需替换为你的训练集路径
    X_train = df_train[continuous_columns]
    X_train_scaled = scaler.transform(X_train)

    explainer = shap.Explainer(model.predict, X_train_scaled[:50])  # 使用部分样本加速
    shap_values = explainer.shap_values(input_scaled)

    # 绘制 SHAP 力图
    shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True)
    plt.savefig("SHAP force plot.png", bbox_inches='tight', dpi=1200)
    st.image("SHAP force plot.png", caption='SHAP Force Plot (Feature Contributions)', use_container_width=True)

    # 提示
    st.markdown("⚙️ **Interpretation:** Positive values indicate an increase in the predicted concentration for that characteristic, while negative values indicate a decrease.")

# ===============================
# 7. 教学提示
# ===============================
st.markdown("---")
st.markdown("💡 **Attention please：**")
st.markdown("""
-This model is a continuous prediction, outputting blood drug concentration (ng/mL).
- '±20%' denotes an empirical confidence interval, within which actual plasma drug concentrations are considered reasonable.
- SHAP values can be used to observe the direction and magnitude of the influence of features on individual predictions.。

""")

