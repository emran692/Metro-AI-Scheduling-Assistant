# train_model.py
import pandas as pd
import lightgbm as lgb
import joblib
import shap
from data_handler import load_and_prepare_data

print("Loading data...")
df = load_and_prepare_data()

if df is not None:
    print("Preparing data for training...")
    df['Target'] = (df['JobCard_Priority'] == 'High').astype(int)
    features = [
        'BrakePad_KM_Since_Change', 'Bogie_KM_Since_Service', 'HVAC_Operating_Hours',
        'Avg_Vibration_Level', 'Max_Brake_Temp_Celsius', 'Min_Cert_Days_to_Expiry'
    ]
    X = df[features]
    y = df['Target']

    print("Training LightGBM model...")
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True)
    lgbm.fit(X, y)

    # --- SAVE THE MODEL ---
    model_filename = 'lgbm_risk_model.joblib'
    joblib.dump(lgbm, model_filename)
    print(f"✅ Model successfully trained and saved to '{model_filename}'")

    # --- NEW: CREATE AND SAVE THE SHAP EXPLAINER ---
    print("Creating SHAP explainer...")
    # SHAP's TreeExplainer is specifically designed for tree-based models like LightGBM
    explainer = shap.TreeExplainer(lgbm)
    explainer_filename = 'shap_explainer.joblib'
    joblib.dump(explainer, explainer_filename)
    print(f"✅ SHAP explainer successfully created and saved to '{explainer_filename}'")
else:
    print("❌ Failed to load data. Script terminated.")
