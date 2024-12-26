# main.py
import argparse
import json
from model_training.train_model import train_random_forest
from model_prediction.predict import predict_contract
from model_validation.validate_model import validate_model

def main():
    parser = argparse.ArgumentParser(description="Pipeline CVCRM ML")
    
    parser.add_argument("--action", type=str, required=True,
                        help="Escolha uma ação: train / validate / predict")
    parser.add_argument("--json_path", type=str, default="data/raw/contracts.json",
                        help="Caminho do arquivo JSON para treinamento ou validação")
    parser.add_argument("--model_path", type=str, default="data/processed/random_forest.pkl",
                        help="Caminho para salvar ou carregar o modelo")
    parser.add_argument("--input_json", type=str, default="{}",
                        help="String JSON de entrada para predição")
    
    args = parser.parse_args()

    if args.action == "train":
        train_random_forest(args.json_path, args.model_path)
    elif args.action == "validate":
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        validate_model(args.model_path, data)
    elif args.action == "predict":
        data = json.loads(args.input_json)
        preds = predict_contract(args.model_path, data)
        print("Predições:", preds)
    else:
        print("Ação inválida. Escolha: train / validate / predict")

if __name__ == "__main__":
    main()
