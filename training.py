import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_ai():
    # Check if CSV exists
    if not os.path.exists('health.csv'):
        print("❌ Error: health.csv not found! Please place the dataset in this folder.")
        return

    try:
        df = pd.read_csv('health.csv')
        # Features: age, cp (chest pain), trestbps (bp), chol (cholesterol)
        # Target: 1=Risk, 0=Healthy
        X = df[['age', 'cp', 'trestbps', 'chol']] 
        y = df['target']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        with open('heart_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✅ AI Training Complete: heart_model.pkl has been created.")
    except Exception as e:
        print(f"❌ Column Error: Ensure your CSV has columns: age, cp, trestbps, chol, target")
        print(f"Specific Error: {e}")

if __name__ == "__main__":
    train_ai()