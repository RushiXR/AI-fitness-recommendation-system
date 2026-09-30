from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
)


X_TRAIN_FILE = (
    PROCESSED_DIR
    / "X_train.csv"
)

X_TEST_FILE = (
    PROCESSED_DIR
    / "X_test.csv"
)

Y_TRAIN_FILE = (
    PROCESSED_DIR
    / "y_train.csv"
)

Y_TEST_FILE = (
    PROCESSED_DIR
    / "y_test.csv"
)


MODEL_FILE = (
    MODELS_DIR
    / "recommended_calories_model.pkl"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Recommended_Calories"


# ============================================================
# MODEL FEATURES
# ============================================================

# BMR is intentionally not used directly.
#
# Recommended calories were derived from Daily Caloric Intake:
#
# Weight Loss -> Daily Calories - 500
# Weight Gain -> Daily Calories + 300
#
# Therefore, we use the underlying user characteristics
# instead of directly supplying the derived BMR value.

MODEL_FEATURES = [
    "Age",
    "Height",
    "Weight",
    "BMI",
    "Sex_Female",
    "Sex_Male",
    "Fitness Goal_Weight Gain",
    "Fitness Goal_Weight Loss",
]


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file}\n\n"
            "Run preprocess.py first."
        )


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("RECOMMENDED CALORIES - LINEAR REGRESSION")
print("========================================")

print("\nLoading training and testing data...")

X_train_full = pd.read_csv(
    X_TRAIN_FILE
)

X_test_full = pd.read_csv(
    X_TEST_FILE
)

y_train = pd.read_csv(
    Y_TRAIN_FILE
)

y_test = pd.read_csv(
    Y_TEST_FILE
)


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in y_train.columns:

    raise ValueError(
        f"\nTarget '{TARGET}' was not found "
        "in y_train.csv."
    )


if TARGET not in y_test.columns:

    raise ValueError(
        f"\nTarget '{TARGET}' was not found "
        "in y_test.csv."
    )


# ============================================================
# CHECK MODEL FEATURES
# ============================================================

missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in X_train_full.columns
]

if missing_features:

    raise ValueError(
        "\nThe following model features are missing:\n"
        + "\n".join(
            f" - {feature}"
            for feature in missing_features
        )
    )


# ============================================================
# SELECT DATA
# ============================================================

X_train = X_train_full[
    MODEL_FEATURES
].copy()

X_test = X_test_full[
    MODEL_FEATURES
].copy()

y_train_target = y_train[
    TARGET
].copy()

y_test_target = y_test[
    TARGET
].copy()


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("\n========================================")
print("MODEL INPUT")
print("========================================")

print("\nFeatures used:")

for feature in MODEL_FEATURES:
    print(f" - {feature}")

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)

print(
    f"Number of input features: "
    f"{len(MODEL_FEATURES)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n========================================")
print("TRAINING MODEL")
print("========================================")

model = LinearRegression()


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train_target
)

print(
    "Linear Regression model trained successfully."
)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(
    y_test_target,
    y_pred
)

mse = mean_squared_error(
    y_test_target,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test_target,
    y_pred
)


print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    f"MAE  : {mae:.2f} kcal"
)

print(
    f"MSE  : {mse:.2f}"
)

print(
    f"RMSE : {rmse:.2f} kcal"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

comparison = pd.DataFrame(
    {
        "Actual": y_test_target.values,
        "Predicted": y_pred,
    }
)

comparison["Difference"] = (
    comparison["Actual"]
    - comparison["Predicted"]
)

comparison["Absolute_Error"] = (
    comparison["Difference"]
    .abs()
)


print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

print(
    comparison
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(MODEL_FILE)

print(
    "\nFile #4 completed successfully."
)