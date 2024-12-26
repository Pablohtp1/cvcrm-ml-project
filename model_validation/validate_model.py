# model_validation/validate_model.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.preprocessing import preprocess_data

def validate_model(model_path, json_data):
    """
    Carrega o modelo e valida com dados de teste em JSON.
    """
    model = joblib.load(model_path)
    df = preprocess_data(json_data)

    if 'target' not in df.columns:
        raise ValueError("Coluna 'target' não encontrada nos dados para validação.")

    X = df.drop(columns=['target'])
    y_true = df['target']

    y_pred = model.predict(X)
    y_proba = None
    
    # Calcula métricas
    acc = accuracy_score(y_true, y_pred)
    try:
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_proba)
    except:
        auc = None
    
    print(f"Accuracy: {acc}")
    print(f"AUC: {auc}" if auc is not None else "AUC não disponível (modelo não suporta predict_proba).")
