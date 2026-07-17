from xgboost import XGBClassifier
import joblib
from src.config import RANDOM_SEED, MODEL_SAVE_PATH

def train_model(X_train, y_train):
    print("[INFO] Training XGBoost Classifier...")
    # Setting random_state for reproducibility
    clf = XGBClassifier(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    clf.fit(X_train, y_train)
    
    print(f"[INFO] Saving model to {MODEL_SAVE_PATH}...")
    joblib.dump(clf, MODEL_SAVE_PATH)
    return clf