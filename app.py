from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('modelo_xgb.pkl')  # Certifique-se de que o arquivo está no mesmo diretório

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 Dados recebidos:", data)

        # Conversão segura dos dados
        features = np.array([[float(data['ma10']), float(data['ma50']), float(data['rsi'])]])

        # Previsão
        prediction = int(model.predict(features)[0])

        # Probabilidades
        probability_up = None
        probability_down_or_neutral = None

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features)[0]
            probability_down_or_neutral = float(probas[0])
            probability_up = float(probas[1])

        return jsonify({
            "prediction": prediction,
            "probability_up": round(probability_up, 4) if probability_up is not None else None,
            "probability_down_or_neutral": round(probability_down_or_neutral, 4) if probability_down_or_neutral is not None else None
        })

    except Exception as e:
        print("❌ Erro interno:", str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "✅ API de previsão de ações está ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
