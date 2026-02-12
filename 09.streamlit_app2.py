import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# --- 1. Robust Path Finder ---
def get_model_path(filename):
    """
    Locates the model file in both Local and HF Space environments.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    potential_paths = [
        os.path.join(script_dir, filename),                 # HF Space / Same folder
        os.path.join(script_dir, "..", "models", filename), # Local (Standard structure)
        os.path.join(os.getcwd(), "models", filename),      # Working dir subfolder
        os.path.join(os.getcwd(), filename)                 # Direct working dir
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    return filename

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    model_filename = "best_model.pkl"
    model_path = get_model_path(model_filename)
    
    if not os.path.exists(model_path):
        st.error(f"🚨 Model file not found: `{model_filename}`")
        st.warning("Please ensure 'best_model.pkl' exists in 'models/' or 'src/' folder.")
        return None
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. Get Feature Names ---
def get_expected_features(model):
    """
    Extracts the feature names the model was trained on.
    """
    try:
        # Check directly on model
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        # Check inside Pipeline steps
        if hasattr(model, 'steps'):
            for _, step_obj in model.steps:
                if hasattr(step_obj, 'feature_names_in_'):
                    return list(step_obj.feature_names_in_)
        return []
    except:
        return []

# --- 4. Crash-Proof Data Alignment ---
def align_data(df, expected_features):
    """
    Forces the input DataFrame to match the model's expected features.
    - Fills missing columns (like ID) with 0.
    - Removes extra columns.
    - Reorders columns to match training data.
    """
    if not expected_features:
        return df
    
    df_aligned = pd.DataFrame()
    
    for col in expected_features:
        # 1. Exact match
        if col in df.columns:
            df_aligned[col] = df[col]
        # 2. Case-insensitive match (id vs ID)
        else:
            match = next((c for c in df.columns if c.lower() == col.lower()), None)
            if match:
                df_aligned[col] = df[match]
            else:
                # 3. MISSING COLUMN -> Fill with 0 (Prevents 'Feature missing' crash)
                # This handles the 'ID' error specifically.
                df_aligned[col] = 0.0
                
    return df_aligned

# --- 5. Output Formatting Helper ---
def format_prediction(pred):
    """
    Formats the prediction label and assigns a color.
    Supports: Up/Down/Neutral or 0/1/2.
    """
    pred_str = str(pred).lower()
    
    # Define colors based on keywords
    if any(x in pred_str for x in ['up', '1', 'buy', 'positive']):
        return pred, "green"
    elif any(x in pred_str for x in ['down', '0', 'sell', 'negative']):
        return pred, "red"
    else:
        return pred, "blue" # Neutral or other

# --- MAIN APPLICATION ---
def main():
    st.title("📈 Tesla Stock Direction Predictor")
    st.markdown("AI-powered classification: **Up**, **Down**, or **Neutral**.")

    # Load Model
    model = load_model()
    if model is None: return

    # Get Features
    expected_features = get_expected_features(model)
    
    # --- TABS ---
    tab1, tab2 = st.tabs(["✍️ Single Data Entry", "📂 Batch Prediction (CSV)"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        st.subheader("Manual Data Entry")
        
        if expected_features:
            # Don't ask user for metadata columns
            ignored_cols = ['id', 'date', 'target', 'class', 'label']
            features_to_ask = [f for f in expected_features if f.lower() not in ignored_cols]
            
            with st.form("single_form"):
                cols = st.columns(3)
                input_data = {}
                
                # Generate input fields
                for i, feature in enumerate(features_to_ask):
                    input_data[feature] = cols[i%3].number_input(feature, value=0.0)
                
                submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Prepare Data
                df_single = pd.DataFrame([input_data])
                df_final = align_data(df_single, expected_features)
                
                try:
                    # Predict
                    pred = model.predict(df_final)[0]
                    
                    # Confidence (if available)
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(df_final)
                        confidence = np.max(probs)
                    
                    # Display
                    label, color = format_prediction(pred)
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown(f"### Prediction: :{color}[{str(pred).upper()}]")
                    with c2:
                        if confidence:
                            st.metric("Confidence Score", f"{confidence:.2%}")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.warning("Could not read model features. Please try CSV upload.")

    # --- TAB 2: CSV UPLOAD ---
    with tab2:
        st.subheader("Batch Prediction")
        st.info("Upload a CSV file containing the technical indicators.")
        
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file and st.button("Run Batch Prediction"):
            try:
                df = pd.read_csv(uploaded_file)
                
                # Align Data (Fixes ID errors automatically)
                X_input = align_data(df, expected_features)
                
                # Predict
                preds = model.predict(X_input)
                
                # Append Results
                results = df.copy()
                results["Predicted_Direction"] = preds
                
                st.success("✅ Prediction Complete!")
                st.dataframe(results.head())
                
                # Download
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="tesla_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
                st.info("Tip: The system automatically handles missing ID columns, so check your data format.")

if __name__ == "__main__":
    main()