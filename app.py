from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('modelo_xgb.pkl')  # Certifique-se que está no mesmo diretório

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # força leitura como JSON, mesmo sem header correto
        data = request.get_json(force=True)
        print("📥 Dados recebidos:", data)

        # verifica presença dos campos
        if not all(k in data for k in ("ma10", "ma50", "rsi")):
            return jsonify({"error": "Campos ma10, ma50, rsi obrigatórios"}), 400

        features = np.array([[data['ma10'], data['ma50'], data['rsi']]])
        prediction = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(features)[0][1])
        else:
            proba = None

        return jsonify({
            "prediction": prediction,
            "probability": round(proba, 4) if proba is not None else None
        })

    except Exception as e:
        print("❌ Erro interno:", str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "✅ API de previsão de ações está ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
