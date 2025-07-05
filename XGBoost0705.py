import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 页面配置 - 设置更宽的布局和页面标题
st.set_page_config(
    page_title="Frailty Predictor for HF Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型和数据
OPTIMAL_THRESHOLD = 0.774
model = joblib.load('XGBoost0705.pkl')
scaler = joblib.load('standardized_data0705.joblib.pkl') 

# 定义选项
Capacity_for_Action_options = {    
    0: 'Bedridden',    
    1: 'Wheelchair dependent',    
    2: 'Ambulatory',    
}

NYHA_Functional_Class_options = {       
    1: 'Class Ⅱ',    
    2: 'Class Ⅲ',    
    3: 'Class Ⅳ',
}

# 定义特征名称
feature_names = [
    "Age", "Capacity_for_Action", "Smoking", "NYHA_Functional_Class", 
    "Thiazide_Diuretics", "Cerebral_Infarction", "Lymphocyte_Percentage",
    "Mean_Corpuscular_Hemoglobin_Concentration", "Albumin",
    "Estimated_Glomerular_Filtration_Rate", "Left_Ventricular_Ejection_Fraction"
]

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .st-bw {
        background-color: white;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background-color: #ffdddd;
        border-left: 5px solid #ff5252;
    }
    .low-risk {
        background-color: #ddffdd;
        border-left: 5px solid #4caf50;
    }
    .feature-section {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .section-title {
        color: #2c3e50;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 主标题
st.title("🏥 Frailty Risk Assessment for Heart Failure Patients")
st.markdown("""
    *This tool predicts the risk of frailty in heart failure patients with acute infections.*  
    *Please fill in the patient's details below and click 'Predict' to get the assessment.*
    """)

# 创建三列布局 (20%, 60%, 20%)
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])

with col2:
    # 使用扩展器组织输入表单
    with st.expander("📋 Patient Information Form", expanded=True):
        with st.form("input_form"):
            # 将输入分成几个部分
            st.markdown("### Demographic Information")
            cols_demo = st.columns(2)
            with cols_demo[0]:
                Age = st.number_input("Age (years)", min_value=1, max_value=150, value=60)
            with cols_demo[1]:
                Capacity_for_Action = st.selectbox(
                    "Mobility Status", 
                    options=list(Capacity_for_Action_options.keys()), 
                    format_func=lambda x: Capacity_for_Action_options[x]
                )
            
            st.markdown("### Clinical Characteristics")
            cols_clinical = st.columns(2)
            with cols_clinical[0]:
                NYHA_Functional_Class = st.selectbox(
                    "NYHA Functional Class", 
                    options=list(NYHA_Functional_Class_options.keys()), 
                    format_func=lambda x: NYHA_Functional_Class_options[x]
                )
                Smoking = st.selectbox(
                    "Smoking Status", 
                    options=[0, 1], 
                    format_func=lambda x: 'Non-smoker' if x == 0 else 'Smoker'
                )
            with cols_clinical[1]:
                Thiazide_Diuretics = st.selectbox(
                    "Thiazide Diuretics Use", 
                    options=[0, 1], 
                    format_func=lambda x: 'No' if x == 0 else 'Yes'
                )
                Cerebral_Infarction = st.selectbox(
                    "History of Cerebral Infarction", 
                    options=[0, 1], 
                    format_func=lambda x: 'No' if x == 0 else 'Yes'
                )
            
            st.markdown("### Laboratory Values")
            cols_lab1 = st.columns(3)
            with cols_lab1[0]:
                Lymphocyte_Percentage = st.number_input(
                    "Lymphocyte Percentage (%)", 
                    min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.1f"
                )
            with cols_lab1[1]:
                Mean_Corpuscular_Hemoglobin_Concentration = st.number_input(
                    "MCHC (g/L)", 
                    min_value=0.0, max_value=1000.0, value=300.0, step=1.0
                )
            with cols_lab1[2]:
                Albumin = st.number_input(
                    "Albumin (g/L)", 
                    min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.1f"
                )
            
            cols_lab2 = st.columns(2)
            with cols_lab2[0]:
                Estimated_Glomerular_Filtration_Rate = st.number_input(
                    "eGFR (%)", 
                    min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f"
                )
            with cols_lab2[1]:
                Left_Ventricular_Ejection_Fraction = st.number_input(
                    "LVEF (%)", 
                    min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f"
                )
            
            submitted = st.form_submit_button("🔍 Predict Frailty Risk", use_container_width=True)

# 准备输入特征
if submitted:
    feature_values = [
        Age, Capacity_for_Action, Smoking, NYHA_Functional_Class, 
        Thiazide_Diuretics, Cerebral_Infarction, Lymphocyte_Percentage,
        Mean_Corpuscular_Hemoglobin_Concentration, Albumin,
        Estimated_Glomerular_Filtration_Rate, Left_Ventricular_Ejection_Fraction
    ]
    
    features = np.array([feature_values])
    
    # 数据预处理
    continuous_features = [Age, Lymphocyte_Percentage, Mean_Corpuscular_Hemoglobin_Concentration, 
                          Albumin, Estimated_Glomerular_Filtration_Rate, Left_Ventricular_Ejection_Fraction]
    categorical_features = [Capacity_for_Action, Smoking, NYHA_Functional_Class, 
                           Thiazide_Diuretics, Cerebral_Infarction]
    
    continuous_features_df = pd.DataFrame(
        np.array(continuous_features).reshape(1, -1), 
        columns=["Age", "Lymphocyte_Percentage", "Mean_Corpuscular_Hemoglobin_Concentration",
                "Albumin", "Estimated_Glomerular_Filtration_Rate", "Left_Ventricular_Ejection_Fraction"]
    )
    
    continuous_features_standardized = scaler.transform(continuous_features_df)
    categorical_features_array = np.array(categorical_features).reshape(1, -1)
    final_features = np.hstack([continuous_features_standardized, categorical_features_array])
    final_features_df = pd.DataFrame(final_features, columns=feature_names)
    
    # 预测
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0
    
    # 显示结果 - 使用中间列
    with col2:
        # 预测结果卡片
        risk_class = "high-risk" if predicted_class == 1 else "low-risk"
        st.markdown(
            f"""
            <div class="prediction-box {risk_class}">
                <h3 style="margin-top:0;">Prediction Results</h3>
                <p style="font-size:24px; font-weight:bold; margin-bottom:0;">
                    Frailty Probability: <span style="color:{'#ff5252' if predicted_class == 1 else '#4caf50'}">{prob_class1:.1%}</span>
                </p>
                <p style="font-size:18px;">
                    Risk Classification: <strong>{'High Risk' if predicted_class == 1 else 'Low Risk'}</strong>
                    (Threshold: {OPTIMAL_THRESHOLD:.0%})
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 临床建议
        advice_box = st.container()
        with advice_box:
            st.markdown("### Clinical Recommendations")
            if predicted_class == 1:
                st.warning("""
                🚨 **High Risk of Frailty Detected**  
                This patient has a significantly elevated risk of frailty. Consider the following actions:
                - Comprehensive geriatric assessment
                - Nutritional supplementation if indicated
                - Supervised physical activity program
                - Medication review for potentially inappropriate medications
                - Close follow-up monitoring
                """)
            else:
                st.success("""
                ✅ **Low Risk of Frailty Detected**  
                While this patient currently shows low risk, consider these preventive measures:
                - Encourage regular physical activity
                - Monitor nutritional status
                - Annual frailty screening
                - Patient education on frailty prevention
                """)
        
        # SHAP解释图
        st.markdown("### Feature Impact Analysis")
        with st.spinner("Generating explanation..."):
            explainer_shap = shap.TreeExplainer(model)
            shap_values = explainer_shap.shap_values(final_features_df)
            
            if isinstance(shap_values, list):
                shap_values_class = shap_values[0]
            else:
                shap_values_class = shap_values
            
            original_feature_values = pd.DataFrame(
                features, 
                columns=feature_names
            )
            
            # 创建更美观的SHAP图
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_class[0], 
                    base_values=explainer_shap.expected_value,
                    data=original_feature_values.iloc[0],
                    feature_names=original_feature_values.columns.tolist()
                ),
                max_display=10,
                show=False
            )
            plt.title("Feature Contribution to Prediction", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("""
            *This waterfall plot shows how each feature contributes to pushing the model's output \
            from the base value (average prediction) to the final prediction. Features in red increase \
            the risk prediction, while features in blue decrease it.*
            """)
        
        # 原始数据展示
        st.markdown("### Input Summary")
        input_data = {
            "Feature": feature_names,
            "Value": feature_values
        }
        st.table(pd.DataFrame(input_data))
