from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('modelo_xgb.pkl')  # Modelo salvo no Colab

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 Dados recebidos:", data)
        features = np.array([[data['ma10'], data['ma50'], data['rsi']]])

        prediction = int(model.predict(features)[0])

        # verificar se o modelo tem o método predict_proba
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(features)[0][1])
        else:
            proba = None

        return jsonify({
            "prediction": prediction,
            "probability": round(proba, 4) if proba is not None else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "API de previsão de ações ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
