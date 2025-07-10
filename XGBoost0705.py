import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Frailty Predictor for HF Patients",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model and data
OPTIMAL_THRESHOLD = 0.774
model = joblib.load('XGBoost0705.pkl')
scaler = joblib.load('standardized_data0705.joblib.pkl') 

# Define options
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

# Feature names
feature_names = [
    "Age", "Capacity_for_Action", "Smoking", "NYHA_Functional_Class", 
    "Thiazide_Diuretics", "Cerebral_Infarction", "Lymphocyte_Percentage",
    "Mean_Corpuscular_Hemoglobin_Concentration", "Albumin",
    "Estimated_Glomerular_Filtration_Rate", "Left_Ventricular_Ejection_Fraction"
]

# Custom CSS for compact layout
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    .st-bw {
        background-color: white;
    }
    
    /* Form styling */
    .stNumberInput, .stSelectbox {
        padding-bottom: 4px;
    }
    div[data-baseweb="input"] {
        margin-bottom: -1rem;
    }
    
    /* Right column styling */
    .right-column {
        font-size: 0.9rem;
    }
    .right-title {
        text-align: left;
        margin-top: 0;
        padding-top: 0;
        font-size: 1.5rem;
    }
    
    /* Prediction box */
    .prediction-box {
        border-radius: 5px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background-color: #ffdddd;
        border-left: 4px solid #ff5252;
    }
    .low-risk {
        background-color: #ddffdd;
        border-left: 4px solid #4caf50;
    }
    
    /* Equal height columns */
    .st-emotion-cache-1cypcdb {
        align-items: stretch;
    }
    
    /* Section headers */
    .section-header {
        font-size: 0.95rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    /* Make columns equal height */
    .column-container {
        display: flex;
        flex-direction: row;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two columns (40%, 60%)
col1, col2 = st.columns([4, 6], gap="medium")

# Right column content
with col2:
    st.markdown("<div class='right-column'>", unsafe_allow_html=True)
    
    # Title and description in right column
    st.markdown("<h1 class='right-title'>🏥 Frailty Risk Assessment for Heart Failure Patients</h1>", unsafe_allow_html=True)
    st.markdown("<p>This tool predicts the risk of frailty in heart failure patients with acute infections.</p>", unsafe_allow_html=True)

# Left column content - input form
with col1:
    with st.container():
        with st.form("input_form"):
            # Demographic Information
            #st.markdown("<div class='section-header'>Demographic Information</div>", unsafe_allow_html=True)
            Age = st.number_input("Age (years)", min_value=1, max_value=150, value=60)
            Capacity_for_Action = st.selectbox(
                "Mobility Status", 
                options=list(Capacity_for_Action_options.keys()), 
                format_func=lambda x: Capacity_for_Action_options[x]
            )
            
            # Clinical Characteristics
            #st.markdown("<div class='section-header'>Clinical Characteristics</div>", unsafe_allow_html=True)
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
            Thiazide_Diuretics = st.selectbox(
                "Thiazide Diuretics Use", 
                options=[0, 1], 
                format_func=lambda x: 'No' if x == 0 else 'Yes'
            )
            Cerebral_Infarction = st.selectbox(
                "Cerebral Infarction History", 
                options=[0, 1], 
                format_func=lambda x: 'No' if x == 0 else 'Yes'
            )
            
            # Laboratory Values
            #st.markdown("<div class='section-header'>Laboratory Values</div>", unsafe_allow_html=True)
            Lymphocyte_Percentage = st.number_input(
                "Lymphocyte Percentage (%)", 
                min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.1f"
            )
            Mean_Corpuscular_Hemoglobin_Concentration = st.number_input(
                "Mean Corpuscular Hemoglobin Concentration (g/L)", 
                min_value=0.0, max_value=1000.0, value=300.0, step=1.0
            )
            Albumin = st.number_input(
                "Albumin (g/L)", 
                min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.1f"
            )
            Estimated_Glomerular_Filtration_Rate = st.number_input(
                "Estimated Glomerular Filtration Rate (%)", 
                min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f"
            )
            Left_Ventricular_Ejection_Fraction = st.number_input(
                "Left Ventricular Ejection Fraction (%)", 
                min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f"
            )
            
            submitted = st.form_submit_button("Predict Frailty Risk", use_container_width=True)

# Prepare input features and show results when submitted
if submitted:
    feature_values = [
        Age, Capacity_for_Action, Smoking, NYHA_Functional_Class, 
        Thiazide_Diuretics, Cerebral_Infarction, Lymphocyte_Percentage,
        Mean_Corpuscular_Hemoglobin_Concentration, Albumin,
        Estimated_Glomerular_Filtration_Rate, Left_Ventricular_Ejection_Fraction
    ]
    
    features = np.array([feature_values])
    
    # Data preprocessing
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
    
    # Prediction
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0
    
    with col2:
        # Prediction results
        risk_class = "high-risk" if predicted_class == 1 else "low-risk"
        st.markdown(
            f"""
            <div class="prediction-box {risk_class}">
                <h3 style='margin-top:0; font-size: 1.1rem;'>Prediction Results</h3>
                <p style="font-size:1rem; font-weight:bold; margin-bottom:0;">
                    Frailty Probability: <span style="color:{'#ff5252' if predicted_class == 1 else '#4caf50'}">{prob_class1:.1%}</span>
                </p>
                <p style="font-size:0.9rem;">
                    Risk Classification: <strong>{'High Risk' if predicted_class == 1 else 'Low Risk'}</strong>
                    (Threshold: {OPTIMAL_THRESHOLD:.0%})
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # SHAP explanation plot
        st.markdown("<div class='section-header'>Feature Impact Analysis</div>", unsafe_allow_html=True)
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
            
            # Create SHAP plot
            fig, ax = plt.subplots(figsize=(9, 5))
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
            plt.title("Feature Contribution to Prediction", fontsize=10, pad=8)
            plt.gcf().set_size_inches(8, 5)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.caption("""
            This plot shows how each feature contributes to the prediction. Features in red increase \
            the risk prediction, while features in blue decrease it.
            """)
        
        # Close right column div
        st.markdown("</div>", unsafe_allow_html=True)
