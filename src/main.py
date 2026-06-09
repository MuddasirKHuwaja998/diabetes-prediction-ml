from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model
with open('Diabaties_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    
    # Extract features (must match training order)
    features = np.array([[
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': round(probability, 2),
        'result': 'Diabetic' if prediction == 1 else 'Not Diabetic'
    })

if __name__ == '__main__':
    app.run(debug=True)