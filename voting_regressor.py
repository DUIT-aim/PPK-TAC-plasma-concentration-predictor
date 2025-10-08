###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
CatBC = joblib.load('CatBoost.pkl')
scaler = joblib.load('scaler.pkl') 

# Streamlit user interface
st.title("thrombocytopenia risk Predictor")

# Define feature names
feature_names = ['ALB','A_G','UREA','Threonine','Phenylacetic_acid','Arginine','Pseudouridine','Linolenic_acid','LysoPC_16_0_0_0','LysoPC_18_3']

ALB = st.number_input("Albumin(U/L):", min_value=10, max_value=80, value=30)
A_G = st.number_input("Albumin/Globulin ratio(%):", min_value=0.0, max_value=3.0, value=1.5)
UREA = st.number_input("Urea(mmol/L):", min_value=1, max_value=50, value=25)
Threonine = st.number_input("Threonine:", min_value=-2.5, max_value=2.0, value=1.0)
Phenylacetic_acid = st.number_input("Phenylacetic acid:", min_value=-2.0, max_value=2.0, value=1.0)
Arginine = st.number_input("Arginine:", min_value=-3.0, max_value=2.0, value=1.0)
Pseudouridine = st.number_input("Pseudouridine:", min_value=-3.0, max_value=3.0, value=0.5)
Linolenic_acid = st.number_input("Linolenic acid:", min_value=-3.5, max_value=3.0, value=0.8)
LysoPC_16_0_0_0 = st.number_input("LysoPC(16:0/0:0):", min_value=-3.0, max_value=2.0, value=0.5)
LysoPC_18_3 = st.number_input("LysoPC(18:3):", min_value=-6.0, max_value=2.0, value=0.6)
#IL10_rs1800896_CT = st.selectbox("IL-10.rs1800896_CT:", options=[1, 2], format_func=lambda x: 'No' if x == 1 else 'Yes')


# 准备输入特征
feature_values = [ALB,A_G,UREA,Threonine,Phenylacetic_acid,Arginine,Pseudouridine,Linolenic_acid,LysoPC_16_0_0_0,LysoPC_18_3]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [ALB,A_G,UREA,Threonine,Phenylacetic_acid,Arginine,Pseudouridine,Linolenic_acid,LysoPC_16_0_0_0,LysoPC_18_3]
#categorical_features=[IL10_rs1800896_CT]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)

# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=['ALB','A_G','UREA','Threonine','Phenylacetic_acid','Arginine',
                                                                          'Pseudouridine','Linolenic_acid','LysoPC_16_0_0_0','LysoPC_18_3'])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
#categorical_features_array = np.array(categorical_features).reshape(1, -1)

# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized])#, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"):    
    OPTIMAL_THRESHOLD = 0.338
    
    # Predict class and probabilities    
    #predicted_class = CatBC.predict(final_features_df)[0]   
    predicted_proba = CatBC.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]  # 类别1的概率

    # 根据最优阈值判断类别
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

    # 显示结果（概率形式更直观）
    st.write(f"**High Exposure Probability:** {prob_class1:.1%}")
    st.write(f"**Decision Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")
    st.write(f"**Predicted Class:** {predicted_class} (1: High risk, 0: Low risk)")
    if predicted_class == 1:  
      advice = (
          
            f"According to the model, you have a high risk of developing thrombocytopenia after taking linezolid. "
          
            f"The model predicts that your probability of having thrombocytopenia is {prob_class1:.1f}%. "
          
            "While this is just an estimate, it suggests that you may be at significant risk. "
          
            "I recommend that you consult a clinician as soon as possible for further evaluation and "
          
            "to ensure you receive an accurate diagnosis and necessary treatment."
          
        )
      st.write(advice)
    else: 
      advice = (
          
            f"According to the model, you have a low risk of developing thrombocytopenia after taking linezolid. "
          
            f"The model predicts that your probability of having thrombocytopenia is {(1 - prob_class1):.1f}%. "
          
            "However, maintaining a healthy lifestyle is still very important. "
          
            "I recommend regular check-ups to monitor your health, "
          
            "and to seek medical advice promptly if you experience any symptoms."
          
        )
      st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")

    # 创建SHAP解释器
    # 假设 x_train 是用于训练模型的特征数据
    df=pd.read_csv('train_lasso.csv',encoding='utf8')
    ytrain=df.LIT
    x_train=df.drop('LIT',axis=1)
    from sklearn.preprocessing import StandardScaler
    continuous_cols = [0,9]
    xtrain = x_train.copy()
    scaler = StandardScaler()
    xtrain.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])

    explainer_shap = shap.KernelExplainer(CatBC.predict_proba, xtrain)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(pd.DataFrame(final_features_df,columns=feature_names))
    
  # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:  
      shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], original_feature_values, matplotlib=True) 
    else: 
      shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], original_feature_values, matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')