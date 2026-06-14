import joblib
import numpy as np

def load_model(model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(input_data, model, scaler):
    input_array = np.array(input_data).reshape(1, -1)
    scaled = scaler.transform(input_array)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    return prediction, round(probability * 100, 2)
