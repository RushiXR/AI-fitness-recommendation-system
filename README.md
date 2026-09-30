# AI Fitness Recommendation System

An AI-powered fitness and nutrition recommendation web application built with **Python, Flask, Pandas, NumPy, and Scikit-learn**.

The application accepts a user's basic fitness information and generates estimated calorie requirements, goal-time estimates, diet recommendations, and workout recommendations using a combination of **Linear Regression** and **Random Forest Classification** models.

---

## Features

- BMI calculation from height and weight
- BMR calculation using the Mifflin-St Jeor equation
- Daily caloric intake estimation
- Recommended calorie estimation based on fitness goal
- Estimated time to reach a reference target
- Diet recommendation using Random Forest Classification
- Workout recommendation using Random Forest Classification
- Data preprocessing using:
  - `StandardScaler`
  - `OneHotEncoder`
  - `ColumnTransformer`
- Flask-based web interface
- Serialized machine-learning models using `joblib`

---

## User Inputs

The application keeps the user interface simple and requires only five inputs:

1. Age
2. Gender
3. Height (cm)
4. Weight (kg)
5. Fitness Goal

The fitness goal currently supports:

- Weight Gain
- Weight Loss

BMI and BMR are calculated automatically from the submitted information.

---

## System Architecture

```text
User
  │
  │ Age, Gender, Height, Weight, Goal
  ▼
Flask Web Application
  │
  ├── BMI Calculation
  ├── BMR Calculation
  │
  ▼
Feature Preprocessing
  │
  ├── StandardScaler
  └── OneHotEncoder
  │
  ▼
Machine Learning Models
  │
  ├── Linear Regression
  │    ├── Daily Caloric Intake
  │    ├── Recommended Calories
  │    └── Time to Goal
  │
  └── Random Forest Classifier
       ├── Diet Recommendation
       └── Workout Recommendation
  │
  ▼
Results Page
