from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
features = joblib.load('features.joblib')

@app.route('/')
def home():
    return "Loan Approval Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query_df = pd.DataFrame([data])
    query_df = query_df[features]
    
    query_scaled = scaler.transform(query_df)
    prediction = model.predict(query_scaled)
    
    return jsonify({'loan_status': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)