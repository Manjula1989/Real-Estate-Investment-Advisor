
import os
import pandas as pd

# Project dataset folder
data_folder = "data"

print("REAL ESTATE INVESTMENT ADVISOR")
print("=" * 40)

# Find CSV files
csv_files = [
    file for file in os.listdir(data_folder)
    if file.lower().endswith(".csv")
]

if not csv_files:
    print("No CSV file found inside the data folder.")
else:
    print("CSV files found:", csv_files)

    file_path = os.path.join(data_folder, csv_files[0])
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
    
print("\n========== DATA QUALITY REPORT ==========")

# Check duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# Check data types
print("\nColumn data types:")
print(df.dtypes)

# Statistical summary
print("\nNumerical summary:")
print(df.describe())

# Unique values in categorical columns
print("\nCategorical column details:")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique()} unique values")

    
# ==========================================
# EXPLORATORY DATA ANALYSIS - VISUALIZATIONS
# ==========================================

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder to save charts
os.makedirs("outputs", exist_ok=True)

# 1. Distribution of property prices
plt.figure(figsize=(9, 5))
sns.histplot(df["Price_in_Lakhs"], kde=True)
plt.title("Distribution of Property Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png")
plt.show()

# 2. Distribution of property sizes
plt.figure(figsize=(9, 5))
sns.histplot(df["Size_in_SqFt"], kde=True)
plt.title("Distribution of Property Sizes")
plt.xlabel("Size (Sq Ft)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.savefig("outputs/size_distribution.png")
plt.show()

# 3. Property size versus price
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    hue="Property_Type"
)
plt.title("Property Size vs Property Price")
plt.tight_layout()
plt.savefig("outputs/size_vs_price.png")
plt.show()

# 4. Average price by property type
plt.figure(figsize=(9, 5))
sns.barplot(
    data=df,
    x="Property_Type",
    y="Price_in_Lakhs",
    estimator="mean"
)
plt.title("Average Property Price by Property Type")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/price_by_property_type.png")
plt.show()

# 5. Average price by top 10 cities
top_cities = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=top_cities,
    x="City",
    y="Price_in_Lakhs"
)
plt.title("Top 10 Cities by Average Property Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/price_by_city.png")
plt.show()

print("\nEDA charts generated successfully!")
print("Check the outputs folder.")



