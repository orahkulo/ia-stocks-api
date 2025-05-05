from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar modelo
model = joblib.load('modelo_xgb.pkl')  # troque pelo seu nome, se diferente

# Endpoint para predição
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Exemplo: ['ma10', 'ma50', 'rsi']
    features = np.array([[data['ma10'], data['ma50'], data['rsi']]])  # ajuste se usar mais/menos features

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0].tolist()

    return jsonify({
        'label': int(prediction),
        'probabilidade': proba
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
