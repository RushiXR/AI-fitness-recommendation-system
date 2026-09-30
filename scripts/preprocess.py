from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "fitness_data.csv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
)

PREPROCESSOR_FILE = (
    MODELS_DIR
    / "preprocessor.pkl"
)


# ============================================================
# INPUT FEATURES
# ============================================================

# These ultimately come from the FIVE user inputs:
#
#   1. Age
#   2. Gender
#   3. Height
#   4. Weight
#   5. Fitness Goal
#
# BMI and BMR are derived automatically.

NUMERIC_FEATURES = [
    "Age",
    "Height",
    "Weight",
    "BMI",
    "BMR",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Fitness Goal",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# TARGETS
# ============================================================

# Regression targets

REGRESSION_TARGETS = [
    "Daily_Caloric_Intake",
    "Recommended_Calories",
    "Time_to_Goal",
]


# Classification targets

CLASSIFICATION_TARGETS = [
    "Diet",
    "Workout_Plan",
]


ALL_TARGETS = (
    REGRESSION_TARGETS
    + CLASSIFICATION_TARGETS
)


# ============================================================
# CHECK INPUT DATASET
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nProcessed dataset not found:\n"
        f"{DATA_FILE}\n\n"
        "Run prepare_data.py first."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("AI FITNESS PROJECT - PREPROCESSING")
print("========================================")

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_FILE)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + ALL_TARGETS
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nThe following required columns are missing:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before_cleaning = len(df)

df = df.dropna(
    subset=required_columns
).copy()

after_cleaning = len(df)

print(
    f"\nRows removed because of missing values: "
    f"{before_cleaning - after_cleaning}"
)

print(
    f"Rows remaining: {after_cleaning}"
)


# ============================================================
# DISPLAY INPUT FEATURES
# ============================================================

print("\n========================================")
print("INPUT FEATURES")
print("========================================")

print("\nNumeric features:")

for feature in NUMERIC_FEATURES:
    print(f" - {feature}")


print("\nCategorical features:")

for feature in CATEGORICAL_FEATURES:
    print(f" - {feature}")


# ============================================================
# DISPLAY TARGETS
# ============================================================

print("\n========================================")
print("TARGETS")
print("========================================")

print("\nRegression targets:")

for target in REGRESSION_TARGETS:
    print(f" - {target}")


print("\nClassification targets:")

for target in CLASSIFICATION_TARGETS:
    print(f" - {target}")


# ============================================================
# CREATE X AND Y
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    ALL_TARGETS
].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# 80% training
# 20% testing
#
# random_state=42 makes the split reproducible.
#
# We stratify using Fitness Goal so that Weight Gain /
# Weight Loss remains approximately balanced in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=X["Fitness Goal"],
)


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)


# ============================================================
# CREATE ONE-HOT ENCODER
# ============================================================

# Newer scikit-learn versions use sparse_output.
#
# The fallback supports older versions as well.

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

except TypeError:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            encoder,
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)


# ============================================================
# FIT PREPROCESSOR
# ============================================================

print(
    "\nFitting preprocessing pipeline..."
)

# IMPORTANT:
#
# Fit only on the training data.
#
# Then use the already-fitted transformer on the test data.

X_train_processed = (
    preprocessor.fit_transform(X_train)
)

X_test_processed = (
    preprocessor.transform(X_test)
)


# ============================================================
# GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print("\n========================================")
print("TRANSFORMED FEATURES")
print("========================================")

print(
    f"Number of model-ready features: "
    f"{len(feature_names)}"
)

for index, feature in enumerate(
    feature_names,
    start=1,
):

    print(
        f"{index:2}. {feature}"
    )


# ============================================================
# CONVERT TO DATAFRAMES
# ============================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index,
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index,
)


# ============================================================
# RESET INDEX
# ============================================================

X_train_processed = (
    X_train_processed
    .reset_index(drop=True)
)

X_test_processed = (
    X_test_processed
    .reset_index(drop=True)
)

y_train = (
    y_train
    .reset_index(drop=True)
)

y_test = (
    y_test
    .reset_index(drop=True)
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

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

FEATURE_NAMES_FILE = (
    PROCESSED_DIR
    / "feature_names.txt"
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

X_train_processed.to_csv(
    X_TRAIN_FILE,
    index=False,
)

X_test_processed.to_csv(
    X_TEST_FILE,
    index=False,
)

y_train.to_csv(
    Y_TRAIN_FILE,
    index=False,
)

y_test.to_csv(
    Y_TEST_FILE,
    index=False,
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE,
)


# ============================================================
# SAVE FEATURE NAMES
# ============================================================

with open(
    FEATURE_NAMES_FILE,
    "w",
    encoding="utf-8",
) as file:

    for feature in feature_names:

        file.write(
            f"{feature}\n"
        )


# ============================================================
# SHOW PROCESSED SHAPES
# ============================================================

print("\n========================================")
print("PROCESSED DATA SHAPES")
print("========================================")

print(
    f"X_train: {X_train_processed.shape}"
)

print(
    f"X_test:  {X_test_processed.shape}"
)

print(
    f"y_train: {y_train.shape}"
)

print(
    f"y_test:  {y_test.shape}"
)


# ============================================================
# DISPLAY TARGET COLUMNS
# ============================================================

print("\n========================================")
print("Y TRAIN COLUMNS")
print("========================================")

for column in y_train.columns:

    print(
        f" - {column}"
    )


# ============================================================
# SHOW SAMPLE TRANSFORMED DATA
# ============================================================

print("\n========================================")
print("PREPROCESSING VERIFICATION")
print("========================================")

print(
    "\nFirst 5 rows of transformed training data:"
)

print(
    X_train_processed
    .head()
    .to_string(index=False)
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")

print("\nFiles created:")

print(
    f" - {X_TRAIN_FILE}"
)

print(
    f" - {X_TEST_FILE}"
)

print(
    f" - {Y_TRAIN_FILE}"
)

print(
    f" - {Y_TEST_FILE}"
)

print(
    f" - {PREPROCESSOR_FILE}"
)

print(
    f" - {FEATURE_NAMES_FILE}"
)

print(
    "\nFile #2 updated successfully."
)