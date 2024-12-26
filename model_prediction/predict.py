# model_prediction/predict.py
import joblib
from utils.preprocessing import preprocess_data

def predict_contract(model_path, json_data):
    """
    Carrega o modelo e faz predição para novas entradas (sem coluna 'target').
    """
    model = joblib.load(model_path)
    df = preprocess_data(json_data)

    # Garante que 'target' não está presente em df
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    
    # Faz a predição
    y_pred = model.predict(df)
    # Para obter probabilidades, caso o modelo suporte:
    # y_proba = model.predict_proba(df)
    
    return y_pred
