import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_india_housing_prices.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("REAL ESTATE INVESTMENT ADVISOR - MODEL TRAINING")
print("=" * 65)

start_time = time.time()

# =========================================================
# 2. LOAD DATASET
# =========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please check the data folder and filename."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Original dataset shape:", df.shape)

required_columns = [
    "City",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Public_Transport_Accessibility"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

# Convert important numeric columns
numeric_columns = [
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "BHK",
    "Size_in_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without valid target values
df = df.dropna(
    subset=["Price_in_Lakhs", "Price_per_SqFt", "City"]
)

df = df[
    (df["Price_in_Lakhs"] > 0) &
    (df["Price_per_SqFt"] > 0)
].copy()

if len(df) < 20:
    raise ValueError("Not enough valid rows to train the models.")

print("Valid dataset shape:", df.shape)

# =========================================================
# 3. CREATE INVESTMENT CLASSIFICATION LABEL
# =========================================================

print("\nCreating investment classification label...")

city_median = df.groupby("City")["Price_per_SqFt"].transform("median")

accessibility = (
    df["Public_Transport_Accessibility"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Transparent project proxy:
# A property is labelled 1 when:
# 1. Its price per sq ft is at or below the city median.
# 2. Public transport accessibility is Medium or High.

df["Good_Investment"] = (
    (df["Price_per_SqFt"] <= city_median) &
    (accessibility.isin(["medium", "high"]))
).astype(int)

print("\nInvestment label distribution:")
print(df["Good_Investment"].value_counts())

if df["Good_Investment"].nunique() < 2:
    raise ValueError(
        "Only one investment class was created. "
        "Please check the dataset and investment label rule."
    )

# =========================================================
# 4. USE A SAMPLE FOR FASTER TRAINING
# =========================================================

MAX_ROWS = 60000

if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)

df = df.reset_index(drop=True)

print("\nRows used for model training:", len(df))

# =========================================================
# 5. DEFINE FEATURES
# =========================================================

classification_features = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_per_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status"
]

regression_features = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_per_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status"
]

# Keep only columns available in the dataset
classification_features = [
    col for col in classification_features if col in df.columns
]

regression_features = [
    col for col in regression_features if col in df.columns
]

if not classification_features or not regression_features:
    raise ValueError("No usable model features were found.")

# =========================================================
# 6. PREPROCESSING FUNCTION
# =========================================================

def build_preprocessor(X):

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    return ColumnTransformer(transformers=transformers)


# =========================================================
# 7. TRAIN CLASSIFICATION MODEL
# =========================================================

print("\n" + "=" * 65)
print("TRAINING INVESTMENT CLASSIFICATION MODEL")
print("=" * 65)

X_class = df[classification_features].copy()
y_class = df["Good_Investment"].copy()

if y_class.value_counts().min() < 2:
    raise ValueError(
        "Each investment class needs at least two rows for training."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

classifier = Pipeline([
    ("preprocessor", build_preprocessor(X_train)),
    (
        "model",
        RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

print("Fitting classification model...")
classifier.fit(X_train, y_train)

class_predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, class_predictions)

print("\nCLASSIFICATION RESULTS")
print("----------------------")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        class_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, class_predictions))

joblib.dump(
    classifier,
    MODEL_DIR / "investment_classifier.pkl"
)

# =========================================================
# 8. TRAIN PROPERTY PRICE REGRESSION MODEL
# =========================================================

print("\n" + "=" * 65)
print("TRAINING PROPERTY PRICE REGRESSION MODEL")
print("=" * 65)

X_reg = df[regression_features].copy()
y_reg = df["Price_in_Lakhs"].copy()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regressor = Pipeline([
    ("preprocessor", build_preprocessor(X_train_r)),
    (
        "model",
        RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

print("Fitting regression model...")
regressor.fit(X_train_r, y_train_r)

price_predictions = regressor.predict(X_test_r)

mae = mean_absolute_error(y_test_r, price_predictions)
rmse = np.sqrt(
    mean_squared_error(y_test_r, price_predictions)
)
r2 = r2_score(y_test_r, price_predictions)

print("\nREGRESSION RESULTS")
print("------------------")
print("MAE:", round(mae, 2), "Lakhs")
print("RMSE:", round(rmse, 2), "Lakhs")
print("R2 Score:", round(r2, 4))

joblib.dump(
    regressor,
    MODEL_DIR / "property_price_regressor.pkl"
)

# =========================================================
# 9. SAVE FEATURE LISTS
# =========================================================

joblib.dump(
    classification_features,
    MODEL_DIR / "classification_features.pkl"
)

joblib.dump(
    regression_features,
    MODEL_DIR / "regression_features.pkl"
)

# =========================================================
# 10. SAVE TRAINING SUMMARY
# =========================================================

summary = {
    "rows_used": len(df),
    "classification_accuracy": accuracy,
    "regression_mae": mae,
    "regression_rmse": rmse,
    "regression_r2": r2,
    "classification_features": classification_features,
    "regression_features": regression_features
}

joblib.dump(
    summary,
    MODEL_DIR / "training_summary.pkl"
)

elapsed = time.time() - start_time

print("\n" + "=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)
print("Models saved in:", MODEL_DIR)
print("Training time:", round(elapsed, 2), "seconds")
print("\nFiles created:")
print("- investment_classifier.pkl")
print("- property_price_regressor.pkl")
print("- classification_features.pkl")
print("- regression_features.pkl")
print("- training_summary.pkl")