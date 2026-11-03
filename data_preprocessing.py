import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 55)
print("REAL ESTATE DATA PREPROCESSING")
print("=" * 55)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "india_housing_prices.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_india_housing_prices.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Remove duplicate records
duplicates = df.duplicated().sum()
df = df.drop_duplicates()

print("Duplicate records removed:", duplicates)

# Remove unnecessary spaces from text columns
text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip()

# Convert numeric columns
numeric_columns = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs",
    "Price_per_SqFt", "Year_Built", "Floor_No",
    "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            )

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset saved successfully:")
print(OUTPUT_PATH)

print("\nDATA PREPROCESSING COMPLETED!")