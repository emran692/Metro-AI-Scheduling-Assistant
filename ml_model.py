# ml_model.py
import joblib
import pandas as pd

class RiskPredictor:
    def __init__(self, model_path='lgbm_risk_model.joblib', explainer_path='shap_explainer.joblib'):
        try:
            self.model = joblib.load(model_path)
            self.explainer = joblib.load(explainer_path) # Load the explainer
        except FileNotFoundError:
            self.model = None
            self.explainer = None
        
        self.features = [
            'BrakePad_KM_Since_Change', 'Bogie_KM_Since_Service', 'HVAC_Operating_Hours',
            'Avg_Vibration_Level', 'Max_Brake_Temp_Celsius', 'Min_Cert_Days_to_Expiry'
        ]

    def predict_risk(self, data):
        if self.model is None: return [0] * len(data)
        X_today = data[self.features]
        return self.model.predict_proba(X_today)[:, 1]

    def get_explanation_df(self, data_row):
        """ NEW: Generates a SHAP explanation for a single row of data. """
        if self.explainer is None: return None
        
        # Ensure data_row is a DataFrame
        if isinstance(data_row, pd.Series):
            data_row = data_row.to_frame().T

        X_row = data_row[self.features]
        
        # Get the raw SHAP values for the positive class (risk of maintenance)
        shap_values = self.explainer.shap_values(X_row)[1]
        
        # Create a DataFrame for easy interpretation
        explanation_df = pd.DataFrame({
            'feature': self.features,
            'feature_value': X_row.iloc[0].values,
            'shap_value': shap_values[0]
        })
        return explanation_df
