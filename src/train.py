import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_config(path="config/config.json"):
    with open(path, "r") as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

def main():
    config = load_config()
    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train, config)
    joblib.dump(model, "model_train.pkl")
    print("✅ Model trained and saved as model_train.pkl")

if __name__ == "__main__":
    main()
