from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('stacking_ensemble_heart_disease.pkl')
scaler = joblib.load('scaler.pkl')

THRESHOLD = 0.4

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = np.array([[
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]])

        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        prediction = int(prob >= THRESHOLD)

        return jsonify({
            "status": "success",
            "probability": round(prob * 100, 2),
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
