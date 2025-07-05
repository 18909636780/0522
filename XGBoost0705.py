###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
OPTIMAL_THRESHOLD = 0.774
model = joblib.load('XGBoost0705.pkl')
scaler = joblib.load('standardized_data0705.joblib.pkl') 

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
st.title("Frailty predictor for heart failure patients with acute infections")

# 创建两列布局
col1, col2 = st.columns([1, 1.5])  # 左列稍窄，右列稍宽

with col1:
    # 输入表单
    with st.form("input_form"):

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
    Lymphocyte_Percentage = st.number_input("Lymphocyte_Percentage(%):", min_value=0.0, max_value=100.0, value=50.0,step=0.1,format="%.2f")
    
    # Mean_Corpuscular_Hemoglobin_Concentration
    Mean_Corpuscular_Hemoglobin_Concentration = st.number_input("Mean_Corpuscular_Hemoglobin_Concentration(g/L):", min_value=0.0, max_value=100.0, value=50.0,step=0.1,format="%.2f")

    # Albumin
    Albumin = st.number_input("Albumin(g/L):", min_value=0.0, max_value=100.0, value=40.0,step=0.1,format="%.2f")

    # Estimated_Glomerular_Filtration_Rate
    Estimated_Glomerular_Filtration_Rate = st.number_input("Estimated_Glomerular_Filtration_Rate(%):", min_value=0.0, max_value=100.0, value=50.0,step=0.1,format="%.2f")

    # Left_Ventricular_Ejection_Fraction
    Left_Ventricular_Ejection_Fraction = st.number_input("Left_Ventricular_Ejection_Fraction(%):", min_value=0.0, max_value=100.0, value=50.0,step=0.1,format="%.2f")


    submitted = st.form_submit_button("Predict")

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


if submitted: 
    with col1:
        # 这里可以留空或放一些其他内容
        pass

    with col2:
        # Predict class and probabilities    
        predicted_proba = model.predict_proba(final_features_df)[0]
        prob_class1 = predicted_proba[1]  # 类别1的概率

        # 根据最优阈值判断类别
        predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

        # 先显示预测结果
        st.subheader("Prediction Results")
        
        # 使用更美观的方式显示结果
        if predicted_class == 1:
            st.error(f"Frailty Probability: {prob_class1:.1%} (High Risk)")
        else:
            st.success(f"Frailty Probability: {prob_class1:.1%} (Low Risk)")
        
        st.write(f"**Risk Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")
        
        # 添加解释性文本
        #st.info(f"""
        #The model predicts a **{prob_class1:.1%} probability** of frailty. 
        #Using the clinically optimized threshold of **{OPTIMAL_THRESHOLD:.0%}**, 
        #this is classified as **{'high risk' if predicted_class == 1 else 'low risk'}**.
        #""")

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

        
        # 添加分隔线
        st.markdown("---")
        
        # 再显示SHAP解释图
        st.subheader("SHAP Explanation")
        
        # 创建SHAP解释器
        explainer_shap = shap.TreeExplainer(model)

        # 获取SHAP值
        shap_values = explainer_shap.shap_values(final_features_df)

        # 确保获取到的shap_values不是None或空值
        if shap_values is None:
            st.error("SHAP values could not be calculated. Please check the model and input data.")
        else:
            # 如果模型返回多个类别的SHAP值（例如分类模型），取相应类别的SHAP值
            if isinstance(shap_values, list):
                shap_values_class = shap_values[0]  # 选择第一个类别的SHAP值
            else:
                shap_values_class = shap_values

            # 将标准化前的原始数据存储在变量中
            original_feature_values = pd.DataFrame(features, columns=["Age","Lymphocyte_Percentage","Mean_Corpuscular_Hemoglobin_Concentration","Albumin","Estimated_Glomerular_Filtration_Rate","Left_Ventricular_Ejection_Fraction"])

            # 创建瀑布图
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap.Explanation(values=shap_values_class[0], 
                                               base_values=explainer_shap.expected_value,
                                               data=original_feature_values.iloc[0],
                                               feature_names=original_feature_values.columns.tolist()))
            
            # 调整图表显示
            plt.tight_layout()
            st.pyplot(fig)