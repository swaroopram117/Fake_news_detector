import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.config import RANDOM_SEED, VECTORIZER_SAVE_PATH, TEST_SIZE_INITIAL, TEST_SIZE_FINAL

def load_and_split_data():
    print("[INFO] Loading True and Fake datasets...")
    # Load both files
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    
    # Add labels: 1 for True, 0 for Fake
    true_df['label'] = 1
    fake_df['label'] = 0
    
    # Combine them
    df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
    
    # Combine title and text
    df['full_text'] = df['title'] + " " + df['text']
    
    X = df['full_text']
    y = df['label']
    
    # Strictly following 70:10:20 split
    print("[INFO] Splitting dataset into 70:10:20 (Train:Val:Test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE_INITIAL, random_state=RANDOM_SEED, stratify=y
    )
    
    # Splitting the 30% into 10% (Val) and 20% (Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE_FINAL, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def extract_features(X_train, X_val, X_test):
    print("[INFO] Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf