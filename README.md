REAL ESTATE
INVESTMENT ADVISOR
Predicting Property Profitability & Future Value
Machine Learning & Data Analytics Project Report
Name: Manjula M
Course: Data Analytics / Business Analytics
Date: 03.11.2026

GitHub Repository
https://github.com/Manjula1989/Real-Estate-Investment-Advisor
 
1. Project Objective
The objective of this project is to develop a Real Estate Investment Advisor using Python, machine learning and Streamlit. The application analyses property characteristics and provides an investment classification together with an estimated property price. It also allows users to explore a five-year property-value scenario using an adjustable annual appreciation assumption.
1.1 Stakeholders and Decisions
Stakeholders: prospective home buyers, real estate investors, property consultants and real estate businesses. Decisions supported: comparing property characteristics, reviewing estimated prices, understanding location and facility factors, and exploring possible future-value scenarios.
1.2 Project Approach
1.	Load and inspect the India housing prices dataset.
2.	Clean the data by reviewing missing values, duplicates, data types and inconsistent categories.
3.	Perform exploratory data analysis to understand price, size, location and property-feature patterns.
4.	Prepare model-ready features through encoding and numerical preprocessing.
5.	Train a classification model for the Good Investment indicator and a regression model for property-price estimation.
6.	Evaluate model performance using suitable classification and regression metrics.
7.	Save trained models and feature-column lists for use by the Streamlit application.
8.	Build an interactive Streamlit application for property input, investment classification and price estimation.
9.	Use an appreciation-rate slider to calculate an illustrative five-year value scenario.
2. Business Questions
•	How are property prices distributed across the dataset?
•	How are property sizes distributed?
•	How does price per square foot vary by property type?
•	Is property size related to the total property price?
•	Are there outliers in property size, price or price per square foot?
•	Which states have higher average price per square foot?
•	How does average property price vary by city?
•	How does property age vary by locality?
•	How are BHK configurations distributed across cities?
•	Which localities have the highest average property prices?
•	Which numerical features have stronger relationships with price?
•	How does the number of nearby schools relate to property price?
•	How does the number of nearby hospitals relate to property price?
•	Does furnishing status influence average property price?
•	Does property-facing direction show price differences?
•	How are properties distributed by owner type?
•	How many properties are listed under each availability status?
•	How does parking availability relate to property price?
•	How do amenities relate to price per square foot?
•	How does public transport accessibility relate to price and investment potential?
3. Dataset Overview
The project brief identifies the source dataset as india_housing_prices.csv. The dataset contains property, location, building, nearby-facility and ownership attributes. The exact row count and column count should be taken from the final CSV used for training and inserted here after verification.
Feature group	Example fields	Purpose
Property and location	State, City, Locality, Property_Type	Describe where the property is and what type it is.
Size and price	BHK, Size_in_SqFt, Price_in_Lakhs, Price_per_SqFt	Represent property size and price information.
Building	Year_Built, Floor_No, Total_Floors, Age_of_Property	Describe building age and floor details.
Facilities	Nearby_Schools, Nearby_Hospitals, Public_Transport_Accessibility	Represent access to local infrastructure.
Property features	Furnished_Status, Parking_Space, Security, Amenities, Facing	Capture features that may influence buyer preference.
Ownership/status	Owner_Type, Availability_Status	Describe listing ownership and availability.

4. Data Preparation and Feature Engineering
4.1 Data Preparation
•	Read the source CSV using pandas and inspect the dataset shape, column names and data types.
•	Review missing values and apply suitable imputation or row-handling rules.
•	Check duplicate records and remove duplicates where appropriate.
•	Convert numerical columns to numeric data types and standardise categorical text values.
•	Review extreme values in price, size and price-per-square-foot fields before model training.
•	Save the prepared dataset as cleaned_india_housing_prices.csv.
4.2 Feature Engineering
Potential engineered features include property age, price per square foot and facility-related indicators. Any derived feature must be calculated consistently for both training data and new property inputs. Features that directly reveal the target value must be excluded from model inputs to prevent target leakage.
4.3 Categorical and Numerical Features
Categorical attributes such as State, City, Locality, Property_Type, Furnished_Status, Public_Transport_Accessibility, Parking_Space, Security, Facing, Owner_Type and Availability_Status require encoding before they can be used by most machine-learning estimators. Numerical attributes such as BHK, Size_in_SqFt, Floor_No, Total_Floors, Age_of_Property, Nearby_Schools and Nearby_Hospitals are retained as numerical inputs after validation.
5. Exploratory Data Analysis (EDA)
EDA is used to understand the dataset before model development. The analysis should include distribution charts, grouped summaries, relationship plots and a correlation heatmap where suitable.
Analysis area	Recommended visual / summary	Question addressed
Price and size	Histograms, box plots, scatter plot	Price distribution, size distribution, outliers and size-price relationship.
Property type	Bar chart / grouped average	Price per square foot by property type.
Location	State and city summary charts	Average price by state and city; locality comparisons.
Building details	Bar chart / grouped median	Property age and BHK distribution.
Facilities	Box plot / grouped average	Price differences by schools, hospitals and transport access.
Property features	Bar charts	Furnishing, facing, parking and amenities comparisons.
Ownership and availability	Count plots	Distribution by owner type and availability status.
Relationships	Correlation heatmap / scatter plot	Relationships among numerical features and property price.

 
Figure 1: Distribution of Property Sizes
The histogram displays the number of properties across different size ranges, measured in square feet. It helps describe the spread of property sizes in the dataset.
6. Machine Learning Model Development
6.1 Investment Classification
The classification task predicts whether a property is labelled as a Good Investment. The project implementation uses a proxy label derived from dataset-based rules. The current application message describes the proxy as being based on city median price per square foot and public transport accessibility. This is a rule-based training label, not a verified measure of actual investment return.
The trained classifier receives the selected classification feature columns and returns a class prediction. The application displays either “Good Investment” or “Not classified as a Good Investment”.
6.2 Property Price Regression
The regression task estimates the current property price in lakhs using the selected property attributes. The trained regression model is loaded by the Streamlit application and produces the estimated price shown in the Property Analysis Results section.
6.3 Five-Year Value Scenario
The application calculates an illustrative future value using compound annual appreciation:
Future Value = Current Estimated Price × (1 + Appreciation Rate / 100)⁵
The appreciation rate is selected by the user through a Streamlit slider. This is a scenario calculation based on an assumption; it is not a direct five-year forecast from a separately trained future-price model. Actual appreciation may differ by locality, market conditions and property type.
7. Model Evaluation
Model evaluation should be documented using the values produced by the final training run. The report intentionally does not invent accuracy or error values. Copy the actual metrics from the terminal output or models/training_summary.pkl before final submission.
Model	Evaluation metrics to report	Purpose
Classification	Accuracy, Precision, Recall, F1-score, Confusion Matrix	Measure how well the classifier identifies the two investment classes.
Regression	MAE, RMSE, R² score	Measure prediction error and how well the model explains price variation.

8. Streamlit Application Overview
The Streamlit application provides a form through which a user can enter property information and request an analysis. The interface is organised into clear groups so that the user can enter location, property, building, nearby-facility and property-feature details.
Application section	Inputs / outputs
Location	State, City and Locality.
Property information	Property Type, BHK, Size in Sq Ft and Price per Sq Ft where included in the current interface.
Building information	Year Built, Furnished Status, Floor Number, Total Floors and Age of Property.
Nearby facilities	Nearby Schools, Nearby Hospitals and Public Transport Accessibility.
Property features	Parking Space, Security, Amenities, Facing, Owner Type and Availability Status.
Analysis results	Investment classification, estimated current property price and five-year value scenario.




























The screenshot below shows the property-entry form and the prediction results generated by the application.
 
Figure 2: Real Estate Investment Advisor – Application Input and Prediction Results
9. Key Insights
The application combines model outputs with user-entered property details. The following are the main types of insights that can be discussed after reviewing the final EDA charts and model outputs:
•	Property price and price per square foot can differ across states, cities, localities and property types.
•	Property size and BHK configuration provide important context when comparing property prices.
•	Nearby schools, hospitals and public transport access can help describe the convenience of a location.
•	Furnishing, parking, security and amenities can contribute to differences in property features and buyer preference.
•	The investment classifier provides a proxy-based category and should be interpreted alongside the underlying property details.
•	The five-year value display changes according to the appreciation-rate assumption selected by the user.

10. Business Recommendations
•	Compare properties within the same city or locality to reduce misleading comparisons between different markets.
•	Review price per square foot together with property size, BHK and property type rather than relying only on total price.
•	Consider nearby schools, hospitals and public transport access as part of a broader location assessment.
•	Check building age, floor details, furnishing, parking, security and amenities before making a purchase decision.
•	Use the investment classification as an initial screening aid, not as a guarantee of profitability.
•	Test multiple appreciation assumptions in the five-year scenario and compare the outcomes before making financial decisions.
11. Limitations and Assumptions
•	The Good Investment label is a proxy label based on dataset rules; it is not based on verified realised investment returns.
•	The regression model estimates current property price. The five-year value is calculated using the appreciation-rate assumption selected in the app.
•	The quality of predictions depends on the coverage, accuracy and representativeness of the supplied housing dataset.
•	The model may be less reliable for localities or property categories that are poorly represented in the training data.
•	The application is an analytical support tool and does not replace a professional property valuation, legal due diligence or financial advice.
•	The dataset does not by itself establish future market conditions, infrastructure changes or property-specific risks.
12. Project Files and Technical Implementation
File / folder	Role
data/india_housing_prices.csv	Original housing dataset.
data/cleaned_india_housing_prices.csv	Preprocessed dataset generated by the data preparation workflow.
data_preprocessing.py	Data cleaning and preprocessing steps.
data_analysis.py	Exploratory data analysis and visualisations.
train_models.py	Training and evaluation of classification and regression models.
models/investment_classifier.pkl	Saved investment classification model.
models/property_price_regressor.pkl	Saved property price regression model.
models/classification_features.pkl	Feature-column list required by the classifier.
models/regression_features.pkl	Feature-column list required by the regressor.
models/training_summary.pkl	Saved training summary, if generated by the final training script.
app.py	Streamlit application for user input and predictions.
outputs/	Generated charts, reports or other analysis outputs, depending on the final run.

12.1 Technologies Used
Python, pandas, NumPy, scikit-learn, joblib, Streamlit and data-visualisation libraries. The project brief also identifies XGBoost and MLflow as relevant technologies; include them in the final implementation description only if they are actually used in the submitted code and run.
12.2 GitHub Repository
The project source code and supporting files are intended to be maintained in the following GitHub repository:
https://github.com/Manjula1989/Real-Estate-Investment-Advisor
Before submission, confirm that the repository contains the Python source files and README. Do not upload the .venv folder, private credentials or unnecessarily large generated files. If the dataset is restricted or too large, document how it can be obtained rather than committing it.
13. Conclusion
The Real Estate Investment Advisor demonstrates an end-to-end data analytics and machine-learning workflow using housing data. The project includes data preparation, exploratory analysis, investment classification, property-price regression and a Streamlit interface. The application helps users review property attributes, view a model-based price estimate and explore an illustrative five-year value scenario. The outputs should be interpreted with the documented proxy-label and appreciation assumptions in mind.
14. Final Submission Checklist
•	☐ Original and cleaned datasets are saved in the project data folder.
•	☐ Data preprocessing script runs without errors.
•	☐ EDA script runs and required charts are saved in outputs.
•	☐ Training script completes and saves both trained models.
•	☐ Actual classification and regression evaluation metrics are added to this report.
•	☐ Streamlit app starts successfully and the Analyse Property button produces results.
•	☐ App screenshots are inserted in this report.
•	☐ Five-year scenario is clearly labelled as assumption-based.
•	☐ GitHub repository contains the final source code and README.
•	☐ No .venv folder, passwords, tokens or private credentials are committed to GitHub.
•	☐ The final report wording matches the code and outputs actually submitted.

Prepared for submission
Manjula M  |  Real Estate Investment Advisor  |  03.11.2026
