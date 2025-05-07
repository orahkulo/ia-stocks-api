from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Carregar modelo treinado
model = joblib.load('modelo_xgb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.info("📥 Dados recebidos: %s", data)

        # Esperadas 12 features
        expected_keys = [
            'ma10', 'ma50', 'rsi',
            'macd_line', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'vix', 'usdbrl', 'selic'
        ]

        # Validação dos campos
        for key in expected_keys:
            if key not in data or data[key] is None:
                raise ValueError(f"Campo ausente ou nulo: {key}")

        # Conversão para float
        features = np.array([[float(data[key]) for key in expected_keys]])

        # Previsão
        prediction = int(model.predict(features)[0])
        proba_up, proba_down = None, None

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features)[0]
            proba_down = float(probas[0])
            proba_up = float(probas[1])

        return jsonify({
            "prediction": prediction,
            "probability_up": round(proba_up, 4),
            "probability_down_or_neutral": round(proba_down, 4)
        })

    except Exception as e:
        logging.error("❌ Erro interno: %s", str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "✅ API de previsão de ações com variáveis técnicas e macroeconômicas está ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
