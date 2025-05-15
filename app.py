from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carrega os modelos treinados para Brasil e EUA
model_br = joblib.load('modelo_xgb_BR.pkl')
model_us = joblib.load('modelo_xgb_US.pkl')

# Define as features esperadas
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

def eh_ativo_brasileiro(ticker):
    return str(ticker).upper().endswith('.SA')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Checa a presença do campo ticker
        if 'ticker' not in input_data or not input_data['ticker']:
            raise ValueError("Campo ausente ou nulo: ticker")

        ticker = input_data['ticker']

        # Checa todos os campos esperados
        for feature in expected_features:
            if feature not in input_data or input_data[feature] is None:
                raise ValueError(f"Campo ausente ou nulo: {feature}")

        # Monta o DataFrame de entrada
        X_input = pd.DataFrame([input_data])[expected_features]

        # Seleciona o modelo conforme o ticker
        if eh_ativo_brasileiro(ticker):
            model = model_br
            mercado = "BR"
        else:
            model = model_us
            mercado = "US"

        # Realiza predição
        pred = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]  # [prob_baixa, prob_alta]

        return jsonify({
            "ticker": ticker,
            "mercado": mercado,
            "prediction": int(pred),              # 0 = baixa, 1 = alta
            "probability_down": round(float(probs[0]), 4),
            "probability_up": round(float(probs[1]), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Pode ser alterado para host='0.0.0.0' em cloud
    app.run(host='0.0.0.0', port=5000)
