from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_FILE = BASE_DIR / "data" / "fitness_data.csv"


# ============================================================
# FIND EXCEL DATASET
# ============================================================

excel_files = list(RAW_DIR.glob("*.xlsx"))

if len(excel_files) == 0:
    raise FileNotFoundError(
        f"\nNo Excel file found inside:\n{RAW_DIR}\n\n"
        "Place the Mendeley Excel dataset inside data/raw/."
    )

if len(excel_files) > 1:
    raise ValueError(
        f"\nMultiple Excel files found inside:\n{RAW_DIR}\n\n"
        + "\n".join(
            f" - {file.name}"
            for file in excel_files
        )
        + "\n\nKeep only one Excel dataset in data/raw/."
    )

RAW_FILE = excel_files[0]


# ============================================================
# LOAD DATASET
# ============================================================

print("========================================")
print("AI FITNESS PROJECT - DATA PREPARATION")
print("========================================")

print("\nDataset file:")
print(RAW_FILE)

print("\nLoading dataset...")

df = pd.read_excel(RAW_FILE)

original_rows = len(df)

print(
    f"Original dataset shape: {df.shape}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print("\nColumns found:")

for column in df.columns:
    print(f" - {column}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ID",
    "Sex",
    "Age",
    "Height",
    "Weight",
    "Hypertension",
    "Diabetes",
    "BMI",
    "Level",
    "Fitness Goal",
    "Fitness Type",
    "Exercises",
    "Equipment",
    "Diet",
    "Recommendation",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# CLEAN SEX
# ============================================================

df["Sex"] = (
    df["Sex"]
    .astype(str)
    .str.strip()
    .str.title()
)

df = df[
    df["Sex"].isin(["Male", "Female"])
].copy()


# ============================================================
# CONVERT NUMERICAL COLUMNS
# ============================================================

df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

df["Height"] = pd.to_numeric(
    df["Height"],
    errors="coerce"
)

df["Weight"] = pd.to_numeric(
    df["Weight"],
    errors="coerce"
)

df["BMI"] = pd.to_numeric(
    df["BMI"],
    errors="coerce"
)


# ============================================================
# REMOVE IMPOSSIBLE VALUES
# ============================================================

df = df[
    (df["Age"] >= 18) &
    (df["Age"] <= 100)
].copy()

df = df[
    (df["Height"] >= 1.2) &
    (df["Height"] <= 2.3)
].copy()

df = df[
    (df["Weight"] >= 30) &
    (df["Weight"] <= 250)
].copy()


# ============================================================
# CALCULATE BMI
# ============================================================

# Height is stored in metres.

df["BMI_Calculated"] = (
    df["Weight"] /
    (df["Height"] ** 2)
)


# ============================================================
# VALIDATE SOURCE BMI
# ============================================================

bmi_difference = (
    df["BMI"] -
    df["BMI_Calculated"]
).abs()

print(
    "\nMaximum difference between source BMI "
    "and calculated BMI:"
)

print(
    bmi_difference.max()
)

print(
    "Rows with BMI difference > 0.1:",
    (bmi_difference > 0.1).sum()
)

print(
    "Rows with BMI difference > 1:",
    (bmi_difference > 1).sum()
)


# Replace source BMI with calculated BMI.
df["BMI"] = df["BMI_Calculated"]

df.drop(
    columns=["BMI_Calculated"],
    inplace=True
)


# ============================================================
# CALCULATE BMR
# ============================================================

# Mifflin-St Jeor equation:
#
# Male:
# BMR = 10W + 6.25H - 5A + 5
#
# Female:
# BMR = 10W + 6.25H - 5A - 161
#
# W = weight in kg
# H = height in cm
# A = age in years

height_cm = df["Height"] * 100

male_bmr = (
    10 * df["Weight"]
    + 6.25 * height_cm
    - 5 * df["Age"]
    + 5
)

female_bmr = (
    10 * df["Weight"]
    + 6.25 * height_cm
    - 5 * df["Age"]
    - 161
)

df["BMR"] = np.where(
    df["Sex"] == "Male",
    male_bmr,
    female_bmr
)


# ============================================================
# CALCULATE DAILY CALORIC INTAKE
# ============================================================

# The source dataset does not contain an activity-level
# measurement.
#
# For this academic project, we assume a moderate activity
# factor of 1.55.
#
# Daily Caloric Intake = BMR * 1.55

ACTIVITY_FACTOR = 1.55

df["Daily_Caloric_Intake"] = (
    df["BMR"] *
    ACTIVITY_FACTOR
)


# ============================================================
# CALCULATE RECOMMENDED CALORIES
# ============================================================

# Project assumptions:
#
# Weight Loss -> 500 kcal/day deficit
# Weight Gain -> 300 kcal/day surplus

fitness_goal_clean = (
    df["Fitness Goal"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Recommended_Calories"] = np.where(
    fitness_goal_clean == "weight loss",
    df["Daily_Caloric_Intake"] - 500,
    df["Daily_Caloric_Intake"] + 300
)

# Prevent unrealistically low values.
df["Recommended_Calories"] = (
    df["Recommended_Calories"]
    .clip(lower=1000)
)


# ============================================================
# CALCULATE TARGET WEIGHT
# ============================================================

# Reference target BMI for the project.

TARGET_BMI = 22

df["Target_Weight"] = (
    TARGET_BMI *
    (df["Height"] ** 2)
)


# ============================================================
# CALCULATE TIME TO GOAL
# ============================================================

# Project assumption:
# 0.5 kg/week rate of weight change.

WEIGHT_CHANGE_RATE = 0.5

weight_difference = (
    df["Weight"] -
    df["Target_Weight"]
).abs()

df["Time_to_Goal"] = (
    weight_difference /
    WEIGHT_CHANGE_RATE *
    7
)

df.drop(
    columns=["Target_Weight"],
    inplace=True
)


# ============================================================
# CREATE CLEAN WORKOUT CLASS
# ============================================================

def classify_workout(exercise_text):
    """
    Convert the source exercise recommendation into
    three canonical workout classes:

        Cardio
        Strength
        Mixed
    """

    text = str(
        exercise_text
    ).strip().lower()

    has_strength = any(
        keyword in text
        for keyword in [
            "squat",
            "deadlift",
            "bench press",
            "overhead press",
            "weightlifting",
            "strength"
        ]
    )

    has_yoga = (
        "yoga" in text
    )

    has_cardio = any(
        keyword in text
        for keyword in [
            "walking",
            "cycling",
            "swimming",
            "running",
            "dancing",
            "cardio"
        ]
    )

    # Strength + Yoga or Strength + Cardio
    # becomes Mixed.

    if (
        (has_strength and has_yoga)
        or
        (has_strength and has_cardio)
    ):
        return "Mixed"

    if has_strength:
        return "Strength"

    if has_cardio:
        return "Cardio"

    return "Mixed"


df["Workout_Plan"] = (
    df["Exercises"]
    .apply(classify_workout)
)


# ============================================================
# CLEAN DATA
# ============================================================

important_columns = [
    "Age",
    "Sex",
    "Height",
    "Weight",
    "BMI",
    "BMR",
    "Fitness Goal",
    "Fitness Type",
    "Diet",
    "Exercises",
    "Equipment",
    "Recommendation",
    "Workout_Plan",
    "Daily_Caloric_Intake",
    "Recommended_Calories",
    "Time_to_Goal",
]

df = df.dropna(
    subset=important_columns
).copy()


# ============================================================
# REMOVE DUPLICATE IDS
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset="ID"
).copy()

after_duplicates = len(df)

duplicates_removed = (
    before_duplicates -
    after_duplicates
)


# ============================================================
# ROUND CALCULATED VALUES
# ============================================================

df["BMI"] = (
    df["BMI"]
    .round(2)
)

df["BMR"] = (
    df["BMR"]
    .round(2)
)

df["Daily_Caloric_Intake"] = (
    df["Daily_Caloric_Intake"]
    .round(2)
)

df["Recommended_Calories"] = (
    df["Recommended_Calories"]
    .round(2)
)

df["Time_to_Goal"] = (
    df["Time_to_Goal"]
    .round(2)
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

final_columns = [
    "ID",
    "Sex",
    "Age",
    "Height",
    "Weight",
    "Hypertension",
    "Diabetes",
    "BMI",
    "Level",
    "Fitness Goal",
    "Fitness Type",
    "Exercises",
    "Equipment",
    "Diet",
    "Recommendation",
    "Workout_Plan",
    "BMR",
    "Daily_Caloric_Intake",
    "Recommended_Calories",
    "Time_to_Goal",
]

df = df[
    final_columns
]


# ============================================================
# SAVE PROCESSED DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("DATA PREPARATION COMPLETE")
print("========================================")

print(
    f"\nOriginal rows: {original_rows}"
)

print(
    f"Final rows:    {len(df)}"
)

print(
    f"Final columns: {len(df.columns)}"
)

print(
    f"Duplicate IDs removed: "
    f"{duplicates_removed}"
)


# ============================================================
# FITNESS GOAL DISTRIBUTION
# ============================================================

print("\n========================================")
print("FITNESS GOAL DISTRIBUTION")
print("========================================")

print(
    df["Fitness Goal"]
    .value_counts()
)


# ============================================================
# WORKOUT PLAN DISTRIBUTION
# ============================================================

print("\n========================================")
print("WORKOUT PLAN DISTRIBUTION")
print("========================================")

print(
    df["Workout_Plan"]
    .value_counts()
)


# ============================================================
# FITNESS TYPE DISTRIBUTION
# ============================================================

print("\n========================================")
print("FITNESS TYPE DISTRIBUTION")
print("========================================")

print(
    df["Fitness Type"]
    .value_counts()
)


# ============================================================
# FITNESS LEVEL DISTRIBUTION
# ============================================================

print("\n========================================")
print("FITNESS LEVEL DISTRIBUTION")
print("========================================")

print(
    df["Level"]
    .value_counts()
)


# ============================================================
# SAMPLE CALCULATED VALUES
# ============================================================

print("\n========================================")
print("SAMPLE CALCULATED VALUES")
print("========================================")

sample_columns = [
    "Age",
    "Sex",
    "Height",
    "Weight",
    "BMI",
    "BMR",
    "Daily_Caloric_Intake",
    "Recommended_Calories",
    "Time_to_Goal",
    "Workout_Plan",
]

print(
    df[
        sample_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# OUTPUT LOCATION
# ============================================================

print("\n========================================")
print("OUTPUT FILE")
print("========================================")

print(
    OUTPUT_FILE
)

print(
    "\nFile #1 updated successfully."
)