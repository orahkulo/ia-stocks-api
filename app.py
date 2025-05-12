from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Carregar modelo treinado
model = joblib.load('modelo_xgb.pkl')

# Listas de features por tipo de ativo
expected_keys_br = [
    'ma10', 'ma50', 'rsi',
    'macd_line', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'vix', 'usdbrl', 'selic',
    'inflacao', 'delta_inflacao', 'delta_inflacao_lag1', 'delta_inflacao_ma3',
    'delta_vix', 'delta_usdbrl', 'delta_selic',
    'delta_vix_lag1', 'delta_usdbrl_lag1', 'delta_selic_lag1',
    'delta_vix_ma3', 'delta_usdbrl_ma3', 'delta_selic_ma3'
]

expected_keys_us = [
    'ma10', 'ma50', 'rsi',
    'macd_line', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'vix',
    'inflacao', 'delta_inflacao', 'delta_inflacao_lag1', 'delta_inflacao_ma3'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.info("📥 Dados recebidos: %s", data)

        # Determina quais chaves são esperadas com base no campo 'ativo_brasileiro'
        is_br = data.get("ativo_brasileiro", True)
        expected_keys = expected_keys_br if is_br else expected_keys_us

        # Verificar se todos os campos esperados estão presentes e não nulos
        for key in expected_keys:
            if key not in data or data[key] is None:
                raise ValueError(f"Campo ausente ou nulo: {key}")

        # Construir vetor de features
        features = np.array([[float(data[key]) for key in expected_keys]])

        # Verificar compatibilidade com o modelo
        if features.shape[1] != model.n_features_in_:
            raise ValueError(f"Quantidade de features incorreta: esperado {model.n_features_in_}, recebido {features.shape[1]}")

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
