from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
model = joblib.load('modelo_xgb.pkl')  # Certifique-se de que este arquivo está no mesmo diretório

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.info("📥 Dados recebidos: %s", data) # Exibe o payload recebido

        # Confirma que todas as features esperadas estão presentes
        expected_keys = [
            'ma10', 'ma50', 'rsi',
            'macd_line', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower'
        ]
        for key in expected_keys:
            if key not in data or data[key] is None:
                raise ValueError(f"Campo ausente ou nulo: {key}")

        # Converte para float com validação
        features = np.array([[float(data[key]) for key in expected_keys]])

        prediction = int(model.predict(features)[0])

        probability_up = None
        probability_down_or_neutral = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability_down_or_neutral = float(proba[0])
            probability_up = float(proba[1])

        return jsonify({
            "prediction": prediction,
            "probability_up": round(probability_up, 4),
            "probability_down_or_neutral": round(probability_down_or_neutral, 4)
        })

    except Exception as e:
        logging.error("❌ Erro interno: %s", str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "✅ API de previsão de ações com indicadores técnicos está ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
