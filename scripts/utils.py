# utils.py
import os
import joblib

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")

def load_model(path):
    return joblib.load(path)
