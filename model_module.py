import pickle
import os

# Path to the model and vectorizer
model_path = 'Model/xgb_model.pkl'
vectorizer_path = 'Model/vectorizer.pkl'

def load_model():
    # Load model and vectorizer
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer
