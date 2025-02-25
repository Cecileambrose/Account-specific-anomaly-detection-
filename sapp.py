import streamlit as st
import pandas as pd
import requests

st.title("Account-Specific Anomaly Detection")

# Upload CSV file
uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.where(pd.notna(df), None)  # Replace NaN with None
    st.write("Preview of Uploaded Data:", df.head())

    if st.button("Detect Anomalies"):
        try:
            response = requests.post("http://127.0.0.1:5000/detect_anomalies", json=df.to_dict(orient="records"))

            st.write("Response Status Code:", response.status_code)  # Print status code
            st.write("Response Text:", response.text)  # Print response text

            if response.status_code == 200:
                anomalies = response.json().get("anomalies", [])
                st.write("Anomalies Detected:", pd.DataFrame(anomalies))
            else:
                st.error(f"Error in detection: {response.status_code}")

        except Exception as e:
            st.error(f"Request failed: {str(e)}")
