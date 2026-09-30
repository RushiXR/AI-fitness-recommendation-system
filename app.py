from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

PREPROCESSOR_FILE = (
    MODELS_DIR / "preprocessor.pkl"
)

DAILY_CALORIES_MODEL_FILE = (
    MODELS_DIR / "daily_caloric_intake_model.pkl"
)

RECOMMENDED_CALORIES_MODEL_FILE = (
    MODELS_DIR / "recommended_calories_model.pkl"
)

TIME_TO_GOAL_MODEL_FILE = (
    MODELS_DIR / "time_to_goal_model.pkl"
)

DIET_MODEL_FILE = (
    MODELS_DIR / "diet_plan_model.pkl"
)

WORKOUT_MODEL_FILE = (
    MODELS_DIR / "workout_plan_model.pkl"
)


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

print("Loading preprocessing pipeline...")

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


# ============================================================
# LOAD ML MODELS
# ============================================================

print("Loading ML models...")

daily_calories_model = joblib.load(
    DAILY_CALORIES_MODEL_FILE
)

recommended_calories_model = joblib.load(
    RECOMMENDED_CALORIES_MODEL_FILE
)

time_to_goal_model = joblib.load(
    TIME_TO_GOAL_MODEL_FILE
)

diet_model = joblib.load(
    DIET_MODEL_FILE
)

workout_model = joblib.load(
    WORKOUT_MODEL_FILE
)


print("All models loaded successfully.")


# ============================================================
# PREPROCESSOR INPUT COLUMNS
# ============================================================

RAW_FEATURE_COLUMNS = [
    "Age",
    "Height",
    "Weight",
    "BMI",
    "BMR",
    "Sex",
    "Fitness Goal",
]


# ============================================================
# MODEL INPUT COLUMNS
# ============================================================

MODEL_FEATURE_COLUMNS = [
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
# CALCULATE BMR
# ============================================================

def calculate_bmr(
    age,
    gender,
    height_cm,
    weight_kg
):
    """
    Calculate BMR using the Mifflin-St Jeor equation.
    """

    if gender == "Male":

        return (
            10 * weight_kg
            + 6.25 * height_cm
            - 5 * age
            + 5
        )

    return (
        10 * weight_kg
        + 6.25 * height_cm
        - 5 * age
        - 161
    )


# ============================================================
# MAKE PREDICTION
# ============================================================

def generate_prediction(
    age,
    gender,
    height_cm,
    weight_kg,
    fitness_goal
):
    """
    Perform the complete prediction pipeline.

    User input
        ↓
    BMI/BMR
        ↓
    preprocessing
        ↓
    five ML models
        ↓
    results
    """

    # ========================================================
    # BMI
    # ========================================================

    height_m = (
        height_cm / 100
    )

    bmi = (
        weight_kg /
        (height_m ** 2)
    )


    # ========================================================
    # BMR
    # ========================================================

    bmr = calculate_bmr(
        age=age,
        gender=gender,
        height_cm=height_cm,
        weight_kg=weight_kg
    )


    # ========================================================
    # CREATE RAW FEATURE DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Height": height_m,
                "Weight": weight_kg,
                "BMI": bmi,
                "BMR": bmr,
                "Sex": gender,
                "Fitness Goal": fitness_goal,
            }
        ]
    )


    # ========================================================
    # APPLY TRAINED PREPROCESSOR
    # ========================================================

    transformed_input = (
        preprocessor.transform(
            input_data[
                RAW_FEATURE_COLUMNS
            ]
        )
    )


    # ========================================================
    # GET FEATURE NAMES
    # ========================================================

    transformed_input = pd.DataFrame(
        transformed_input,
        columns=(
            preprocessor
            .get_feature_names_out()
        )
    )


    # ========================================================
    # SELECT FEATURES USED BY OUR MODELS
    # ========================================================

    model_input = (
        transformed_input[
            MODEL_FEATURE_COLUMNS
        ]
    )


    # ========================================================
    # REGRESSION PREDICTIONS
    # ========================================================

    daily_calories = (
        daily_calories_model
        .predict(model_input)[0]
    )

    recommended_calories = (
        recommended_calories_model
        .predict(model_input)[0]
    )

    time_to_goal = (
        time_to_goal_model
        .predict(model_input)[0]
    )


    # ========================================================
    # CLASSIFICATION PREDICTIONS
    # ========================================================

    diet_plan = (
        diet_model
        .predict(model_input)[0]
    )

    workout_plan = (
        workout_model
        .predict(model_input)[0]
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "goal": fitness_goal,

        "bmi": round(
            float(bmi),
            2
        ),

        "bmr": round(
            float(bmr),
            2
        ),

        "daily_caloric_intake": round(
            float(daily_calories),
            2
        ),

        "recommended_calories": round(
            float(recommended_calories),
            2
        ),

        "time_to_goal_days": round(
            max(
                0,
                float(time_to_goal)
            ),
            2
        ),

        "diet_plan": str(
            diet_plan
        ),

        "workout_plan": str(
            workout_plan
        ),
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ====================================================
        # READ INPUT
        # ====================================================

        if request.is_json:

            data = request.get_json()

            age = data.get("age")
            gender = data.get("gender")
            height_cm = data.get("height")
            weight_kg = data.get("weight")
            fitness_goal = data.get("goal")

        else:

            age = request.form.get("age")
            gender = request.form.get("gender")
            height_cm = request.form.get("height")
            weight_kg = request.form.get("weight")
            fitness_goal = request.form.get("goal")


        # ====================================================
        # CHECK MISSING INPUTS
        # ====================================================

        values = [
            age,
            gender,
            height_cm,
            weight_kg,
            fitness_goal,
        ]

        if any(
            value is None
            or str(value).strip() == ""
            for value in values
        ):

            error_message = (
                "All five inputs are required."
            )

            if request.is_json:

                return jsonify(
                    {
                        "status": "error",
                        "message": error_message,
                    }
                ), 400

            return error_message, 400


        # ====================================================
        # CONVERT NUMBERS
        # ====================================================

        age = int(age)

        height_cm = float(
            height_cm
        )

        weight_kg = float(
            weight_kg
        )


        # ====================================================
        # NORMALIZE TEXT
        # ====================================================

        gender = (
            str(gender)
            .strip()
            .title()
        )

        fitness_goal = (
            str(fitness_goal)
            .strip()
        )


        # ====================================================
        # VALIDATE GENDER
        # ====================================================

        if gender not in [
            "Male",
            "Female",
        ]:

            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Gender must be Male "
                            "or Female."
                        ),
                    }
                ),
                400,
            )


        # ====================================================
        # VALIDATE GOAL
        # ====================================================

        if fitness_goal not in [
            "Weight Gain",
            "Weight Loss",
        ]:

            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Goal must be Weight Gain "
                            "or Weight Loss."
                        ),
                    }
                ),
                400,
            )


        # ====================================================
        # VALIDATE AGE
        # ====================================================

        if not 18 <= age <= 100:

            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Age must be between "
                            "18 and 100."
                        ),
                    }
                ),
                400,
            )


        # ====================================================
        # VALIDATE HEIGHT
        # ====================================================

        if not 120 <= height_cm <= 230:

            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Height must be between "
                            "120 and 230 cm."
                        ),
                    }
                ),
                400,
            )


        # ====================================================
        # VALIDATE WEIGHT
        # ====================================================

        if not 30 <= weight_kg <= 250:

            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            "Weight must be between "
                            "30 and 250 kg."
                        ),
                    }
                ),
                400,
            )


        # ====================================================
        # GENERATE PREDICTIONS
        # ====================================================

        result = generate_prediction(
            age=age,
            gender=gender,
            height_cm=height_cm,
            weight_kg=weight_kg,
            fitness_goal=fitness_goal,
        )


        # ====================================================
        # JSON REQUEST
        # ====================================================

        if request.is_json:

            return jsonify(
                {
                    "status": "success",
                    "input": {
                        "age": result["age"],
                        "gender": result["gender"],
                        "height_cm": result[
                            "height_cm"
                        ],
                        "weight_kg": result[
                            "weight_kg"
                        ],
                        "goal": result["goal"],
                    },
                    "derived": {
                        "bmi": result["bmi"],
                        "bmr": result["bmr"],
                    },
                    "predictions": {
                        "daily_caloric_intake": result[
                            "daily_caloric_intake"
                        ],
                        "recommended_calories": result[
                            "recommended_calories"
                        ],
                        "time_to_goal_days": result[
                            "time_to_goal_days"
                        ],
                        "diet_plan": result[
                            "diet_plan"
                        ],
                        "workout_plan": result[
                            "workout_plan"
                        ],
                    },
                }
            )


        # ====================================================
        # HTML FORM REQUEST
        # ====================================================

        return render_template(
            "result.html",
            result=result
        )


    # ========================================================
    # INPUT ERROR
    # ========================================================

    except ValueError as error:

        return (
            jsonify(
                {
                    "status": "error",
                    "message": (
                        f"Invalid input: {error}"
                    ),
                }
            ),
            400,
        )


    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as error:

        print(
            "\nPrediction error:",
            error
        )

        return (
            jsonify(
                {
                    "status": "error",
                    "message": (
                        "An unexpected error occurred."
                    ),
                }
            ),
            500,
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("AI FITNESS APPLICATION")
    print("========================================")

    print(
        "\nStarting Flask server..."
    )

    print(
        "Open http://127.0.0.1:5000 "
        "in your browser."
    )

    app.run(
        debug=True
    )