from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Carregar o modelo treinado
model = joblib.load('modelo_xgb.pkl')

# ✅ Definir as features esperadas
expected_features = [
    'ma10', 'ma50', 'rsi',
    'macd_line', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'vix', 'usdbrl', 'selic',
    'inflacao_index', 'inflacao_mom', 'inflacao_yoy',
    'delta_vix', 'delta_usdbrl', 'delta_selic', 'delta_inflacao_index',
    'delta_vix_lag1', 'delta_usdbrl_lag1', 'delta_selic_lag1', 'delta_inflacao_index_lag1',
    'delta_vix_ma3', 'delta_usdbrl_ma3', 'delta_selic_ma3', 'delta_inflacao_index_ma3'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # ✅ Validar presença de todos os campos
        for feature in expected_features:
            if feature not in input_data or input_data[feature] is None:
                raise ValueError(f"Campo ausente ou nulo: {feature}")

        # ✅ Transformar em DataFrame
        X_input = pd.DataFrame([input_data])[expected_features]

        # ✅ Realizar predição
        pred = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]

        return jsonify({
            "prediction": int(pred),  # 0: queda, 1: neutro, 2: alta
            "probability_down": round(float(probs[0]), 4),
            "probability_neutral": round(float(probs[1]), 4),
            "probability_up": round(float(probs[2]), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
