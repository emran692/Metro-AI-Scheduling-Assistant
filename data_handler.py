# data_handler.py
import pandas as pd

def load_and_prepare_data(file_path='kmrl_enhanced_dataset.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None # Return None if the file isn't found
    
    # Convert date columns to the correct format
    date_cols = ['Date', 'RollingStock_Cert_Expiry', 'Signal_Cert_Expiry', 'Telecom_Cert_Expiry']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Engineer the 'Min_Cert_Days_to_Expiry' feature
    df['Min_Cert_Days_to_Expiry'] = df[date_cols[1:]].apply(
        lambda row: (row - df.loc[row.name, 'Date']).dt.days, axis=1
    ).min(axis=1)
    
    return df