from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask API is running!"

# Load the dataset
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

@app.route("/detect_anomalies", methods=["POST"])
def detect_anomalies():
    data = request.get_json()
    df = pd.DataFrame(data)

    # Ensure columns exist
    required_features = ["amount_value", "booking_year", "booking_month", "booking_day",
                         "booking_weekday", "category_name", "creditDebitIndicator", "merchant_name"]

    for col in required_features:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    anomalies = []

    for account_id, group in df.groupby("accountId"):
        if len(group) < 10:
            continue

        X = group[required_features]

        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        group["anomaly_flag"] = iso_forest.fit_predict(X)
        group["anomaly_flag"] = group["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

        anomalies.extend(group[group["anomaly_flag"] == 1].to_dict(orient="records"))

    return jsonify({"anomalies": anomalies})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

