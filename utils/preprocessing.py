# utils/preprocessing.py
import pandas as pd

def preprocess_data(json_data):
    """
    Recebe um dicionário em formato JSON, converte para DataFrame e aplica transformações.
    """
    try:
        # Converte o dicionário para um DataFrame
        df = pd.DataFrame([json_data])
        
        # Ajusta tipos (exemplo: converte todas as colunas para float)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Trata valores nulos substituindo por zero ou mediana (exemplo)
        df.fillna(df.median(), inplace=True)
        
        return df
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return None
