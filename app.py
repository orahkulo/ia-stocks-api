from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('modelo_xgb.pkl')  # Certifique-se de que este arquivo está no mesmo diretório

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 Dados recebidos:", data)

        # Lista de features esperadas, na ordem correta
        features_ordem = [
            'ma10', 'ma50', 'rsi',
            'macd_line', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower'
        ]

        # Extrai as features na ordem correta e força conversão para float
        features = np.array([[float(data[f]) for f in features_ordem]])

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
        print("❌ Erro interno:", str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "✅ API de previsão de ações com indicadores técnicos está ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
