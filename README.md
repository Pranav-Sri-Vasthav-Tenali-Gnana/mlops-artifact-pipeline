# Digit Classification – MLOps Assignment

This repository contains a complete MLOps workflow for classifying handwritten digits (0–9) using Logistic Regression and the `sklearn.datasets.load_digits` dataset. The workflow demonstrates parameterized model training, unit testing, inference, and multi-job GitHub Actions CI pipelines with artifact passing.

## Project Structure

```

mlops-artifact-pipeline/
├── src/
│   ├── train.py                 # Model training logic
│   ├── inference.py            # Model inference logic
│   └── utils.py                # (Optional) shared utilities
├── config/
│   └── config.json             # Hyperparameter configuration
├── tests/
│   └── test\_train.py           # Pytest unit tests
├── .github/workflows/
│   ├── train.yml               # Training pipeline
│   ├── test.yml                # Testing pipeline
│   └── inference.yml           # Multi-job pipeline: test → train → inference
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── model\_train.pkl             # Trained model artifact (generated)

```

## Setup Instructions

### 1. Clone the Repository

```

git clone [https://github.com/your-username/mlops-artifact-pipeline.git](https://github.com/your-username/mlops-artifact-pipeline.git)
cd mlops-artifact-pipeline

```

### 2. Create and Activate Environment

Using conda:
```

conda create -n mlops-env python=3.10 -y
conda activate mlops-env

```

### 3. Install Dependencies

```

pip install -r requirements.txt

```

## Running the Project

### Classification Branch

Implements model training using Logistic Regression. The model is saved as `model_train.pkl`.

To run locally:

```

git checkout classification
python src/train.py

```

### Test Branch

Contains unit tests for configuration, model structure, and accuracy.

To run locally:

```

git checkout test
pytest tests/

```

### Inference Branch

Performs inference using the trained model and implements a multi-stage CI/CD pipeline.

To run locally:

```

git checkout inference
python src/inference.py

```

## Continuous Integration with GitHub Actions

GitHub Actions workflows are configured as follows:

- `train.yml`: Trains model and uploads it as an artifact (triggered on `classification` branch).
- `test.yml`: Runs pytest (triggered on `test` branch).
- `inference.yml`: Multi-job pipeline that:
  1. Runs tests
  2. Trains model
  3. Downloads model artifact and runs inference

These workflows ensure reproducibility and enforce correctness via CI.

## Notes

- Hyperparameters are loaded from `config/config.json`. No hardcoded values are allowed.
- Model artifact `model_train.pkl` is passed between jobs using `upload-artifact` and `download-artifact`.
- The repository uses the following linear branching pattern as required:

```

main → classification → test → inference

```

- All Git operations were performed via command line only.

## Author

Pranav Sri Vasthav Tenali Gnana  
G24AI1114  
Indian Institute of Technology Jodhpur