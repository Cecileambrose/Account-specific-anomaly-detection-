import streamlit as st
import pandas as pd
import requests

# Load the dataset
df_cleaned = pd.read_csv("account_specific_transactions.csv")

df = df_cleaned.where(pd.notna(df_cleaned), None)  # Replace NaN with None (JSON friendly)
response = requests.post("https://account-specific-anomaly-detection-1.onrender.com", json=df.to_dict(orient="records"))



st.title("Account-Specific Anomaly Detection")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:", df.head())

    if st.button("Detect Anomalies"):
        response = requests.post("https://account-specific-anomaly-detection-1.onrender.com", json=df.to_dict(orient="records"))

    
    st.write("Response Status Code:", response.status_code)  # Print status code
    st.write("Response Text:", response.text)  # Print response text

    if response.status_code == 200:
        anomalies = response.json().get("anomalies", [])
        st.write("Anomalies Detected:", pd.DataFrame(anomalies))
    else:
        st.error(f"Error in detection: {response.status_code}")
