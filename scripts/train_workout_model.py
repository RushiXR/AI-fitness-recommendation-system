from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
    / "workout_plan_model.pkl"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Workout_Plan"


# ============================================================
# MODEL FEATURES
# ============================================================

# The model uses information derived from the original
# five user inputs:
#
#   Age
#   Gender
#   Height
#   Weight
#   Fitness Goal
#
# BMI and BMR are derived features.

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
print("WORKOUT PLAN - RANDOM FOREST CLASSIFIER")
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
# SELECT MODEL DATA
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
# CLEAN TARGET VALUES
# ============================================================

y_train_target = (
    y_train_target
    .astype(str)
    .str.strip()
)

y_test_target = (
    y_test_target
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE INVALID TARGET VALUES
# ============================================================

valid_train_mask = (
    y_train_target.notna()
    & (y_train_target != "")
    & (y_train_target != "nan")
)

valid_test_mask = (
    y_test_target.notna()
    & (y_test_target != "")
    & (y_test_target != "nan")
)


X_train = X_train.loc[
    valid_train_mask
].reset_index(drop=True)

y_train_target = y_train_target.loc[
    valid_train_mask
].reset_index(drop=True)


X_test = X_test.loc[
    valid_test_mask
].reset_index(drop=True)

y_test_target = y_test_target.loc[
    valid_test_mask
].reset_index(drop=True)


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
# DISPLAY CLASSES
# ============================================================

classes = sorted(
    y_train_target.unique()
)


print("\n========================================")
print("WORKOUT CLASSES")
print("========================================")

print(
    f"Number of classes: {len(classes)}"
)

for index, class_name in enumerate(
    classes,
    start=1
):

    print(
        f"{index}. {class_name}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("TRAINING CLASS DISTRIBUTION")
print("========================================")

print(
    y_train_target.value_counts()
)


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

print("\n========================================")
print("TRAINING MODEL")
print("========================================")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train_target
)

print(
    "Random Forest classifier trained successfully."
)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test_target,
    y_pred
)


print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test_target,
        y_pred,
        labels=classes,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

matrix = confusion_matrix(
    y_test_target,
    y_pred,
    labels=classes
)

print(matrix)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

comparison = pd.DataFrame(
    {
        "Actual": y_test_target.values,
        "Predicted": y_pred,
    }
)


print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

print(
    comparison
    .head(15)
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

print(
    MODEL_FILE
)

print(
    "\nFile #7 updated successfully."
)