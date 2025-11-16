# 1_-_Daily_Schedule.py
import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from data_handler import load_and_prepare_data
from ml_model import RiskPredictor
from optimizer import generate_schedule

st.set_page_config(page_title="KMRL Scheduling Assistant", page_icon="🚆", layout="wide")

# Load data and model
df = load_and_prepare_data()
predictor = RiskPredictor()

st.title("🚆 KMRL AI Scheduling Assistant")
st.sidebar.header("Controls")

# (Sidebar and data preparation code remains the same)
selected_date = st.sidebar.date_input("Select a date for scheduling", value=df['Date'].max() + datetime.timedelta(days=1))

if st.sidebar.button("Generate Optimal Schedule", type="primary"):
    daily_data = df[df['Date'] == pd.to_datetime(selected_date)].copy()
    if daily_data.empty:
        # Use last known data for future dates
        daily_data = df.loc[df.groupby('Train_ID')['Date'].idxmax()].copy()
        daily_data['Date'] = pd.to_datetime(selected_date)
    
    daily_data.reset_index(drop=True, inplace=True)
    daily_data['Min_Cert_Days_to_Expiry'] = daily_data[['RollingStock_Cert_Expiry', 'Signal_Cert_Expiry', 'Telecom_Cert_Expiry']].apply(
        lambda row: (row - daily_data.loc[row.name, 'Date']).dt.days, axis=1
    ).min(axis=1)

    risk_scores = predictor.predict_risk(daily_data)
    daily_data['Predicted_Risk'] = risk_scores
    schedule_df = generate_schedule(daily_data, daily_data['Predicted_Risk'])

    st.header(f"Recommended Schedule for {selected_date.strftime('%Y-%m-%d')}")
    if schedule_df is not None:
        final_display = pd.merge(schedule_df, daily_data, on='Train_ID')
        st.dataframe(final_display[['Train_ID', 'Assigned_Status', 'Reason']], use_container_width=True)

        st.subheader("Risk Prediction Details")
        for index, row in final_display.iterrows():
            with st.expander(f"**{row['Train_ID']}** - Predicted Risk: **{row['Predicted_Risk']:.2%}**"):
                st.write(f"#### Why was this risk score predicted?")
                
                explanation_df = predictor.get_explanation_df(row)
                
                if explanation_df is not None:
                    # Filter for features with a significant impact
                    explanation_df['abs_shap'] = explanation_df['shap_value'].abs()
                    significant_df = explanation_df[explanation_df['abs_shap'] > 0.01].sort_values(by='abs_shap', ascending=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Factors INCREASING Risk 🔺")
                        pos_factors = significant_df[significant_df['shap_value'] > 0]
                        if pos_factors.empty:
                            st.write("None")
                        else:
                            for _, factor in pos_factors.iterrows():
                                st.markdown(f"- **{factor['feature'].replace('_', ' ')}** is {factor['feature_value']:.0f}")

                    with col2:
                        st.markdown("##### Factors DECREASING Risk 🔽")
                        neg_factors = significant_df[significant_df['shap_value'] < 0]
                        if neg_factors.empty:
                            st.write("None")
                        else:
                             for _, factor in neg_factors.iterrows():
                                st.markdown(f"- **{factor['feature'].replace('_', ' ')}** is {factor['feature_value']:.0f}")
                else:
                    st.warning("Could not generate an explanation.")
    else:
        st.error("Could not find an optimal solution.")
