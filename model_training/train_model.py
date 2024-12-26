# model_training/train_model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess_data

def train_random_forest(json_path, output_model_path):
    """
    Lê dados JSON, treina um RandomForestClassifier e salva o modelo treinado.
    """
    # Exemplo: leitura de um conjunto de dados
    with open(json_path, 'r') as f:
        json_data = f.read()
    
    # Converte a string JSON para dicionário
    import json
    data_dict = json.loads(json_data)

    # Pré-processa os dados
    df = preprocess_data(data_dict)

    # Exemplo simples: vamos supor que existe uma coluna 'target' no JSON
    X = df.drop(columns=['target'], errors='ignore')
    y = df['target'] if 'target' in df.columns else None

    if y is None:
        raise ValueError("Não foi encontrada a coluna 'target' nos dados.")

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instancia e treina o modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Salva o modelo treinado
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"Modelo RandomForest salvo em: {output_model_path}")
