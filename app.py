from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route("/detect_anomalies", methods=["POST"])
def detect_anomalies():
    # Check if the CSV file exists
    if not os.path.exists("account_specific_transactions.csv"):
        return jsonify({"error": "CSV file not found on the server"}), 500

    # Load dataset inside the function to avoid crashes
    df_cleaned = pd.read_csv("account_specific_transactions.csv")

    # Preprocess data
    df_cleaned["bookingDateTime"] = pd.to_datetime(df_cleaned["bookingDateTime"], errors="coerce")
    df_cleaned.fillna({"merchant_name": "Unknown"}, inplace=True)
    df_cleaned.drop(columns=["merchant_logo", "merchant_merchantCategoryCode"], errors='ignore', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ["accountProductType", "providerId", "transactionDescription",
                        "category_group", "category_name", "creditDebitIndicator", "merchant_name"]

    for col in categorical_cols:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoders[col] = le

    df_cleaned["booking_year"] = df_cleaned["bookingDateTime"].dt.year
    df_cleaned["booking_month"] = df_cleaned["bookingDateTime"].dt.month
    df_cleaned["booking_day"] = df_cleaned["bookingDateTime"].dt.day
    df_cleaned["booking_weekday"] = df_cleaned["bookingDateTime"].dt.weekday
    df_cleaned["amount_value"] = df_cleaned["amount_value"].apply(lambda x: np.log1p(x))

    # Get input JSON data
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data received"}), 400
    
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    # Ensure required columns exist
    required_features = ["amount_value", "booking_year", "booking_month", "booking_day",
                         "booking_weekday", "category_name", "creditDebitIndicator", "merchant_name"]

    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        return jsonify({"error": f"Missing columns: {', '.join(missing_cols)}"}), 400

    anomalies = []

    for account_id, group in df.groupby("accountId"):
        if len(group) < 10:
            continue  # Skip small groups

        X = group[required_features]

        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        group["anomaly_flag"] = iso_forest.fit_predict(X)
        group["anomaly_flag"] = group["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

        anomalies.extend(group[group["anomaly_flag"] == 1].to_dict(orient="records"))

    return jsonify({"anomalies": anomalies})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
