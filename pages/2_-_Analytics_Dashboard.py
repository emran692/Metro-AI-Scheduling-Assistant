# pages/2_-_Analytics_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Fleet Health & Analytics Dashboard")

# --- Use the dataframe from the session state ---
# This ensures that if a user uploads a new dataset on the main page,
# the analytics dashboard will update accordingly.
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("No dataset loaded. Please go to the 'Scheduling Assistant' page to load a dataset.")
else:
    df = st.session_state.df
    st.markdown("This dashboard reflects the currently loaded dataset.")

    latest_data = df.loc[df.groupby('Train_ID')['Date'].idxmax()]
    today = pd.to_datetime(datetime.date.today())
    
    # --- Row 1: Key Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fleet Size", f"{latest_data['Train_ID'].nunique()} Trains")
    active_maintenance = latest_data[latest_data['JobCard_Open'] == True]['Train_ID'].nunique()
    col2.metric("Trains Currently in Maintenance", f"{active_maintenance} Trains")
    expiring_soon = latest_data[
        ((latest_data['RollingStock_Cert_Expiry'] - today).dt.days < 30) |
        ((latest_data['Signal_Cert_Expiry'] - today).dt.days < 30) |
        ((latest_data['Telecom_Cert_Expiry'] - today).dt.days < 30)
    ].shape[0]
    col3.metric("Certificates Expiring in 30 Days", f"{expiring_soon} Certs")

    st.divider()

    # --- Row 2: Charts ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mileage Balancing Overview")
        fig_mileage = px.bar(
            latest_data.sort_values(by='Bogie_KM_Since_Service', ascending=False),
            x='Train_ID', y='Bogie_KM_Since_Service', title="Bogie Kilometers Since Last Service",
            color='Bogie_KM_Since_Service', color_continuous_scale='reds'
        )
        st.plotly_chart(fig_mileage, use_container_width=True)

    with col2:
        st.subheader("Common Maintenance Issues")
        maintenance_jobs = df[df['JobCard_Open'] == True]['JobCard_Type'].value_counts()
        fig_jobs = px.pie(
            values=maintenance_jobs.values, names=maintenance_jobs.index,
            title="Frequency of Job Card Types", hole=0.3
        )
        st.plotly_chart(fig_jobs, use_container_width=True)

    st.divider()

    # --- Row 3: Upcoming Certificate Expirations Table ---
    st.subheader("Upcoming Certificate Expirations (Next 90 Days)")
    for cert in ['RollingStock', 'Signal', 'Telecom']:
        latest_data[f'{cert}_Days_Left'] = (latest_data[f'{cert}_Cert_Expiry'] - today).dt.days
    expiring_df = latest_data[
        (latest_data['RollingStock_Days_Left'] < 90) |
        (latest_data['Signal_Days_Left'] < 90) |
        (latest_data['Telecom_Days_Left'] < 90)
    ].sort_values(by='RollingStock_Days_Left')
    st.dataframe(
        expiring_df[['Train_ID', 'RollingStock_Days_Left', 'Signal_Days_Left', 'Telecom_Days_Left']],
        use_container_width=True
    )
