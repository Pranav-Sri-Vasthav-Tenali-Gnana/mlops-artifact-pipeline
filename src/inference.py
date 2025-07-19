import joblib
from sklearn.datasets import load_digits

def load_model(path="model_train.pkl"):
    return joblib.load(path)

def run_inference(model):
    digits = load_digits()
    X = digits.data
    predictions = model.predict(X)
    return predictions

def main():
    model = load_model()
    predictions = run_inference(model)
    print("🔮 First 10 Predictions:", predictions[:10])

if __name__ == "__main__":
    main()
