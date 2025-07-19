import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import json
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.train import load_config, train_model

CONFIG_PATH = "config/config.json"

def test_config_file_exists():
    assert os.path.exists(CONFIG_PATH), "Config file is missing"

def test_config_valid():
    config = load_config(CONFIG_PATH)
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training_and_type():
    config = load_config(CONFIG_PATH)
    digits = load_digits()
    X_train, _, y_train, _ = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")

def test_model_accuracy():
    config = load_config(CONFIG_PATH)
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, config)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.8, f"Accuracy too low: {acc}"
