###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('MLP.pkl')
scaler = joblib.load('standardized_data.joblib2.pkl') 

# Define feature options
Capacity_for_Action_options = {    
    0: 'bedridden',    
    1: 'wheelchair_dependent',    
    2: 'ambulatory',    
}

NYHA_Functional_Class_options = {       
    1: 'Ⅱ',    
    2: 'Ⅲ',    
    3: 'Ⅳ',
}


# Define feature names
feature_names = ["Age", "Capacity_for_Action", "Smoking", "NYHA_Functional_Class", "Thiazide_Diuretics", "Cerebral_Infarction", "Lymphocyte_Percentage","Mean_Corpuscular_Hemoglobin_Concentration","Albumin","Estimated_Glomerular_Filtration_Rate","Left_Ventricular_Ejection_Fraction"]

# Streamlit user interface
st.title("Frailty_Predictor")

# Age
Age = st.number_input("Age:", min_value=1, max_value=150, value=60)

# Capacity_for_Action
Capacity_for_Action = st.selectbox("Capacity_for_Action:", options=list(Capacity_for_Action_options.keys()), format_func=lambda x: Capacity_for_Action_options[x])

# Smoking
Smoking = st.selectbox("Smoking:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# NYHA_Functional_Class
NYHA_Functional_Class= st.selectbox("NYHA_Functional_Class:", options=list(NYHA_Functional_Class_options.keys()), format_func=lambda x: NYHA_Functional_Class_options[x])

# Thiazide_Diuretics
Thiazide_Diuretics = st.selectbox("Thiazide_Diuretics:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Cerebral_Infarction
Cerebral_Infarction = st.selectbox("Cerebral_Infarction:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Lymphocyte_Percentage
Lymphocyte_Percentage = st.number_input("Lymphocyte_Percentage(%):", min_value=1, max_value=100, value=50)

# Mean_Corpuscular_Hemoglobin_Concentration
Mean_Corpuscular_Hemoglobin_Concentration = st.number_input("Mean_Corpuscular_Hemoglobin_Concentration(g/L):", min_value=1, max_value=100, value=50)

# Albumin
Albumin = st.number_input("Albumin(g/L):", min_value=1, max_value=100, value=40)

# Estimated_Glomerular_Filtration_Rate
Estimated_Glomerular_Filtration_Rate = st.number_input("Estimated_Glomerular_Filtration_Rate(%):", min_value=1, max_value=100, value=50)

# Left_Ventricular_Ejection_Fraction
Left_Ventricular_Ejection_Fraction = st.number_input("Left_Ventricular_Ejection_Fraction(%):", min_value=1, max_value=100, value=50)



# 准备输入特征
feature_values = [Age, Capacity_for_Action, Smoking, NYHA_Functional_Class, Thiazide_Diuretics, Cerebral_Infarction, Lymphocyte_Percentage,Mean_Corpuscular_Hemoglobin_Concentration,Albumin,Estimated_Glomerular_Filtration_Rate,Left_Ventricular_Ejection_Fraction]

features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [Age,Lymphocyte_Percentage,Mean_Corpuscular_Hemoglobin_Concentration,Albumin,Estimated_Glomerular_Filtration_Rate,Left_Ventricular_Ejection_Fraction]
categorical_features=[Capacity_for_Action, Smoking, NYHA_Functional_Class, Thiazide_Diuretics, Cerebral_Infarction]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=["Age","Lymphocyte_Percentage","Mean_Corpuscular_Hemoglobin_Concentration","Albumin","Estimated_Glomerular_Filtration_Rate","Left_Ventricular_Ejection_Fraction"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(0: No Disease,1: Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
         advice = (            
                f"According to our model, you have a high risk of frailty. "            
                f"The model predicts that your probability of having frailty is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model, you have a low risk of frailty. "            
                f"The model predicts that your probability of not having frailty is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

# SHAP Explanation
st.subheader("SHAP Waterfall Plot Explanation")

# 读取数据并预处理
df = pd.read_csv('test_data0312.csv', encoding='utf8')
ytest = df.Frailty
xtest = df.drop('Frailty', axis=1)

# 定义特征和标签
continuous_cols = ['Age', 'Lymphocyte_Percentage', 'Mean_Corpuscular_Hemoglobin_Concentration', 
                  'Albumin', 'Estimated_Glomerular_Filtration_Rate', 'Left_Ventricular_Ejection_Fraction']

xtest = df[continuous_cols]

# 初始化标准化器
scaler = StandardScaler()

# 对测试数据进行转换（使用训练数据的参数）
xtest_standard = xtest.copy()
xtest_standard[continuous_cols] = scaler.transform(xtest[continuous_cols])

# 创建SHAP解释器
explainer_shap = shap.KernelExplainer(model.predict_proba, shap.sample(xtest_standard, 100))  # 使用100个样本作为背景数据

# 准备要解释的样本数据
sample_to_explain = pd.DataFrame(final_features_df, columns=feature_names)

# 获取模型预测概率和预测类别
predicted_proba = model.predict_proba(sample_to_explain)
predicted_class = model.predict(sample_to_explain)[0]  # 获取预测类别

# 获取SHAP值
shap_values = explainer_shap.shap_values(sample_to_explain)

# 创建瀑布图
plt.figure()
if predicted_class == 1:
    # 对于预测类别1
    shap.plots._waterfall.waterfall_legacy(explainer_shap.expected_value[1], 
                                         shap_values[1][0], 
                                         feature_names=feature_names,
                                         max_display=10)
else:
    # 对于预测类别0
    shap.plots._waterfall.waterfall_legacy(explainer_shap.expected_value[0], 
                                         shap_values[0][0], 
                                         feature_names=feature_names,
                                         max_display=10)

# 添加图表标题
plt.title(f"SHAP Waterfall Plot (Predicted Class: {predicted_class})")

# 保存并显示图像
plt.tight_layout()
plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_waterfall_plot.png", caption='SHAP Waterfall Plot Explanation')
