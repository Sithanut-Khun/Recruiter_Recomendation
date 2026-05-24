import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

@st.cache_resource
def load_models():
    model_dir = 'models'
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder_target.pkl'))
    onehot_encoder = joblib.load(os.path.join(model_dir, 'onehot_encoder.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'robust_scaler.pkl'))  # This scaler was fit on 3 numerical columns only
    
    models = {}
    model_files = ['LightGBM.pkl', 'XGBoost.pkl', 'Logistic_Regression.pkl', 'Random_Forest.pkl']
    model_names = ['LightGBM', 'XGBoost', 'Logistic Regression', 'Random Forest']
    for file, name in zip(model_files, model_names):
        path = os.path.join(model_dir, file)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return feature_names, label_encoder, onehot_encoder, scaler, models

def preprocess_input(categorical_values, numeric_values, onehot_encoder, scaler, feature_names):
    """
    Preprocess user input exactly as in training:
    X = np.hstack([X_cat_encoded, X_num_scaled])
    """
    # Step 1: Create categorical DataFrame with the exact column names used in training
    cat_df = pd.DataFrame([{
        'Business_Unit_Anon': categorical_values['Business_Unit_Anon'],
        'JobType_Anon': categorical_values['JobType_Anon'],
        'JobLevel_Anon': categorical_values['JobLevel_Anon'],
        'JobPriority': categorical_values['JobPriority'],
        'job_group': categorical_values['job_group'],
        'ReasonToRecruit_Anon': categorical_values['ReasonToRecruit_Anon']
    }])
    
    # Step 2: One-hot encode categorical features
    cat_encoded = onehot_encoder.transform(cat_df)  # shape (1, n_cat_features)
    
    # Step 3: Scale numerical features (scaler was fit on these 3 columns only)
    num_df = pd.DataFrame([[
        numeric_values['NumberOfOpening'],
        numeric_values['HireDuration'],
        numeric_values['TimeToFill']
    ]], columns=['NumberOfOpening', 'HireDuration', 'TimeToFill'])
    num_scaled = scaler.transform(num_df)  # shape (1, 3)
    
    # Step 4: Combine categorical (first) then numerical (last) - matches training order
    X_combined = np.hstack([cat_encoded, num_scaled])  # shape (1, total_features)
    
    return X_combined

def get_categories_from_encoder(onehot_encoder):
    """Extract category options from onehot encoder"""
    categories_dict = {}
    for i, feature in enumerate(onehot_encoder.feature_names_in_):
        categories_dict[feature] = onehot_encoder.categories_[i].tolist()
    return categories_dict

def main():
    st.set_page_config(page_title="Recruiter Recommender", layout="wide")
    st.title("📌 AI Recruiter Recommender System")
    st.markdown("Enter job details to get **top 3 recruiters** with highest probability.")
    
    with st.spinner("Loading models..."):
        try:
            feature_names, label_encoder, onehot_encoder, scaler, models = load_models()
            st.success(f"✅ Loaded {len(models)} models")
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()
    
    # Get dropdown options from the one-hot encoder
    categories_dict = get_categories_from_encoder(onehot_encoder)
    
    with st.form("prediction_form"):
        st.subheader("📋 Job Information")
        col1, col2 = st.columns(2)
        
        with col1:
            business_unit = st.selectbox("Business Unit", categories_dict['Business_Unit_Anon'])
            job_type = st.selectbox("Job Type", categories_dict['JobType_Anon'])
            job_level = st.selectbox("Job Level", categories_dict['JobLevel_Anon'])
        
        with col2:
            job_priority = st.selectbox("Job Priority", categories_dict['JobPriority'])
            job_group = st.selectbox("Job Group", categories_dict['job_group'])
            reason_to_recruit = st.selectbox("Reason to Recruit", categories_dict['ReasonToRecruit_Anon'])
        
        st.subheader("🔢 Quantitative Metrics")
        col3, col4, col5 = st.columns(3)
        with col3:
            num_openings = st.number_input("Number of Openings", min_value=1, value=1, step=1)
        with col4:
            hire_duration = st.number_input("Hire Duration (days)", min_value=1, value=30, step=1)
        with col5:
            time_to_fill = st.number_input("Time to Fill (days)", min_value=1, value=45, step=1)
        
        model_choice = st.selectbox("Choose Prediction Model", list(models.keys()))
        submitted = st.form_submit_button("🔍 Recommend Recruiters", use_container_width=True)
    
    if submitted:
        # Prepare values with exact column names used in training
        categorical_values = {
            'Business_Unit_Anon': business_unit,
            'JobType_Anon': job_type,
            'JobLevel_Anon': job_level,
            'JobPriority': job_priority,
            'job_group': job_group,
            'ReasonToRecruit_Anon': reason_to_recruit
        }
        numeric_values = {
            'NumberOfOpening': num_openings,
            'HireDuration': hire_duration,
            'TimeToFill': time_to_fill
        }
        
        try:
            X_input = preprocess_input(categorical_values, numeric_values,
                                       onehot_encoder, scaler, feature_names)
            st.success("✅ Input processed successfully")
        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            st.stop()
        
        # Get prediction
        model = models[model_choice]
        try:
            probabilities = model.predict_proba(X_input)[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()
        
        # Get top 3 recruiters
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_probs = probabilities[top3_idx]
        top3_recruiters = label_encoder.inverse_transform(top3_idx)
        
        # Display results
        st.subheader("🏆 Top 3 Recommended Recruiters")
        
        # Bar chart
        chart_df = pd.DataFrame({"Recruiter": top3_recruiters, "Probability": top3_probs})
        st.bar_chart(chart_df.set_index("Recruiter"), use_container_width=True)
        
        # Detailed cards
        for i, (recruiter, prob) in enumerate(zip(top3_recruiters, top3_probs), 1):
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #667eea {prob*100}%, #e0e0e0 {prob*100}%);
                        padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h2 style="margin: 0;">{i}. {recruiter}</h2>
                <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{prob:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Full distribution expander
        with st.expander("📊 View All Recruiters' Probabilities"):
            full_df = pd.DataFrame({
                "Recruiter": label_encoder.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            st.bar_chart(full_df.head(20).set_index("Recruiter"), use_container_width=True)
            st.dataframe(full_df.style.format({'Probability': '{:.2%}'}), use_container_width=True)

if __name__ == "__main__":
    main()