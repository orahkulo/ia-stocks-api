from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('modelo_ia_stocks.pkl')  # Certifique-se de que o .pkl está na mesma pasta

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[data['ma10'], data['ma50'], data['rsi']]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "API de previsão de ações ativa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
