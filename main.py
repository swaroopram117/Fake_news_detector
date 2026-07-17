from src.preprocess import load_and_split_data, extract_features
from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    print("=== Fake News Detection Pipeline Started ===")
    
    # 1. Load and Split Data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # 2. Extract Features
    X_train_feat, X_val_feat, X_test_feat = extract_features(X_train, X_val, X_test)
    
    # 3. Train Model
    model = train_model(X_train_feat, y_train)
    
    # 4. Evaluate Model on Validation Set
    evaluate_model(model, X_val_feat, y_val, dataset_name="Validation")
    
    # 5. Evaluate Model on Test Set
    evaluate_model(model, X_test_feat, y_test, dataset_name="Test")
    
    print("=== Pipeline Completed Successfully ===")