import os
import numpy as np

RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# No single DATA_PATH needed now, we will load True and Fake directly
MODEL_SAVE_PATH = 'data/xgboost_model.pkl'
VECTORIZER_SAVE_PATH = 'data/tfidf_vectorizer.pkl'

TEST_SIZE_INITIAL = 0.30 
TEST_SIZE_FINAL = 0.6666