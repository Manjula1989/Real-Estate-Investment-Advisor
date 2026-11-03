import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CLASSIFIER_PATH = MODEL_DIR / "investment_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "property_price_regressor.pkl"

CLASS_FEATURES_PATH = MODEL_DIR / "classification_features.pkl"
REG_FEATURES_PATH = MODEL_DIR / "regression_features.pkl"

# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 3. CUSTOM COLOURFUL DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f2fe);
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    background: linear-gradient(90deg, #4338ca, #0891b2);
    color: white;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.18);
}

.main-title h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.main-title p {
    color: #e0f2fe;
    font-size: 16px;
}

.section-title {
    color: #312e81;
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    min-height: 190px;
    border: 1px solid #dbeafe;
    box-shadow: 0 7px 20px rgba(30, 64, 175, 0.10);
}

.warning-card {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border-left: 7px solid #d97706;
}

.success-card {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border-left: 7px solid #16a34a;
}

.result-label {
    color: #475569;
    font-size: 14px;
    font-weight: bold;
}

.result-value {
    color: #4338ca;
    font-size: 30px;
    font-weight: 900;
    margin-top: 20px;
}

.future-card {
    background: linear-gradient(110deg, #4338ca, #2563eb, #0891b2);
    color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 15px;
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.20);
}

.future-card h2,
.future-card h3,
.future-card p {
    color: white;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #4f46e5, #0891b2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #3730a3, #0e7490);
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #dbeafe;
    box-shadow: 0 6px 20px rgba(30, 64, 175, 0.08);
}

.footer {
    text-align: center;
    color: #475569;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():

    required_files = [
        CLASSIFIER_PATH,
        REGRESSOR_PATH,
        CLASS_FEATURES_PATH,
        REG_FEATURES_PATH
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {file_path.name}"
            )

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    class_features = joblib.load(CLASS_FEATURES_PATH)
    reg_features = joblib.load(REG_FEATURES_PATH)

    return classifier, regressor, class_features, reg_features


try:
    classifier, regressor, class_features, reg_features = load_models()

except Exception as error:
    st.error(f"Unable to load trained models: {error}")
    st.warning("Please run python train_models.py first.")
    st.stop()

# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.markdown("""
<div class="main-title">
    <h1>🏠 Real Estate Investment Advisor</h1>
    <p>
        Smart Property Analysis | Investment Classification |
        Property Price Estimation
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the property details below to estimate its current "
    "market price and evaluate its investment potential."
)

# =========================================================
# 6. PROPERTY INPUT FORM
# =========================================================

st.markdown(
    '<div class="section-title">📍 Property Information</div>',
    unsafe_allow_html=True
)

with st.form("property_form"):

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # COLUMN 1: LOCATION AND PROPERTY INFORMATION
    # -----------------------------------------------------

    with col1:

        st.subheader("📍 Location")

        state = st.text_input(
            "State",
            value="Karnataka"
        )

        city = st.text_input(
            "City",
            value="Bangalore"
        )

        locality = st.text_input(
            "Locality",
            value="Whitefield"
        )

        st.subheader("🏢 Property Information")

        property_type = st.selectbox(
            "Property Type",
            [
                "Apartment",
                "Villa",
                "Independent House",
                "Plot"
            ]
        )

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2
        )

        size = st.number_input(
            "Size (Sq Ft)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100
        )

        price_per_sqft = st.number_input(
            "Price per Sq Ft (₹)",
            min_value=100.0,
            max_value=1000000.0,
            value=1750.0,
            step=50.0
        )

    # -----------------------------------------------------
    # COLUMN 2: BUILDING INFORMATION
    # -----------------------------------------------------

    with col2:

        st.subheader("🏗️ Building Information")

        year_built = st.number_input(
            "Year Built",
            min_value=1950,
            max_value=2026,
            value=2023
        )

        furnished = st.selectbox(
            "Furnished Status",
            [
                "Furnished",
                "Semi-Furnished",
                "Unfurnished"
            ]
        )

        floor_no = st.number_input(
            "Floor Number",
            min_value=0,
            max_value=100,
            value=2
        )

        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=100,
            value=10
        )

        age = st.number_input(
            "Age of Property",
            min_value=0,
            max_value=100,
            value=3
        )

        st.subheader("🚉 Nearby Facilities")

        schools = st.number_input(
            "Nearby Schools",
            min_value=0,
            max_value=100,
            value=3
        )

        hospitals = st.number_input(
            "Nearby Hospitals",
            min_value=0,
            max_value=100,
            value=2
        )

        transport = st.selectbox(
            "Public Transport Accessibility",
            ["Low", "Medium", "High"]
        )

    # -----------------------------------------------------
    # COLUMN 3: PROPERTY FEATURES
    # -----------------------------------------------------

    with col3:

        st.subheader("✨ Property Features")

        parking = st.selectbox(
            "Parking Space",
            ["Yes", "No"]
        )

        security = st.selectbox(
            "Security",
            ["Yes", "No"]
        )

        amenities = st.text_input(
            "Amenities",
            value="Gym, Swimming Pool, Lift"
        )

        facing = st.selectbox(
            "Facing",
            ["East", "West", "North", "South"]
        )

        owner_type = st.selectbox(
            "Owner Type",
            ["Owner", "Dealer", "Builder"]
        )

        availability = st.selectbox(
            "Availability Status",
            [
                "Ready_to_Move",
                "Under_Construction"
            ]
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Analyse Property",
        use_container_width=True
    )

# =========================================================
# 7. RUN PREDICTIONS
# =========================================================

if submitted:

    property_data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_per_SqFt": price_per_sqft,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability
    }

    input_df = pd.DataFrame([property_data])

    try:

        # Prepare model inputs using saved feature lists
        class_input = input_df[class_features]
        reg_input = input_df[reg_features]

        # Classification prediction
        investment_prediction = int(
            classifier.predict(class_input)[0]
        )

        # Classification confidence, if available
        probabilities = classifier.predict_proba(class_input)[0]
        confidence = float(max(probabilities) * 100)

        # Regression prediction
        predicted_price = float(
            regressor.predict(reg_input)[0]
        )

        if not pd.notna(predicted_price) or predicted_price <= 0:
            raise ValueError(
                "The model returned an invalid property price."
            )

        # Save result so the five-year slider can update
        # without requiring another form submission.
        st.session_state["property_result"] = {
            "investment_prediction": investment_prediction,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "property_data": property_data
        }

        st.session_state["analysis_completed"] = True

    except Exception as error:

        st.error(f"Prediction error: {error}")
        st.session_state["analysis_completed"] = False

# =========================================================
# 8. DISPLAY RESULTS
# =========================================================

if st.session_state.get("analysis_completed", False):

    result = st.session_state["property_result"]

    investment_prediction = result["investment_prediction"]
    confidence = result["confidence"]
    predicted_price = result["predicted_price"]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Property Analysis Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    # -----------------------------------------------------
    # INVESTMENT RECOMMENDATION
    # -----------------------------------------------------

    with result_col1:

        if investment_prediction == 1:

            st.markdown(f"""
            <div class="result-card success-card">
                <div class="result-label">
                    ✅ INVESTMENT RECOMMENDATION
                </div>
                <h2 style="color:#166534;">
                    Good Investment
                </h2>
                <p style="color:#166534;">
                    The trained model classified this property
                    as a potential good investment.
                </p>
                <b style="color:#166534;">
                    Model confidence: {confidence:.2f}%
                </b>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card warning-card">
                <div class="result-label">
                    ⚠️ INVESTMENT RECOMMENDATION
                </div>
                <h2 style="color:#92400e;">
                    Not classified as a Good Investment
                </h2>
                <p style="color:#92400e;">
                    Review the property features before
                    making an investment decision.
                </p>
                <b style="color:#92400e;">
                    Model confidence: {confidence:.2f}%
                </b>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # CURRENT PRICE
    # -----------------------------------------------------

    with result_col2:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                💰 ESTIMATED CURRENT PROPERTY PRICE
            </div>
            <div class="result-value">
                ₹ {predicted_price:,.2f} Lakhs
            </div>
            <p>
                Estimated by the trained regression model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # FIVE-YEAR PROPERTY VALUE SCENARIO
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Five-Year Property Value Scenario</div>',
        unsafe_allow_html=True
    )

    appreciation_rate = st.slider(
        "Select Assumed Annual Appreciation Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
        key="appreciation_rate"
    )

    future_price = predicted_price * (
        1 + appreciation_rate / 100
    ) ** 5

    st.markdown(f"""
    <div class="future-card">
        <h3>🚀 Estimated Property Value After 5 Years</h3>
        <h1>₹ {future_price:,.2f} Lakhs</h1>
        <p>
            Assumed annual appreciation:
            <b>{appreciation_rate:.1f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # PROPERTY SUMMARY
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("📋 Property Details Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Property Size", f"{size:,} Sq Ft")
        st.metric("BHK", bhk)

    with summary_col2:
        st.metric("Price per Sq Ft", f"₹ {price_per_sqft:,.2f}")
        st.metric("Nearby Schools", schools)

    with summary_col3:
        st.metric("Nearby Hospitals", hospitals)
        st.metric("Transport Accessibility", transport)

    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    st.info(
        "Investment classification is based on a proxy label "
        "created from the dataset's city median price per square "
        "foot and public transport accessibility. The estimated "
        "current price is a model output. The five-year value is "
        "a scenario based on your selected appreciation rate, "
        "not a direct five-year forecast from the trained model. "
        "These results are not a guarantee of investment returns."
    )

else:

    st.info(
        "Enter the property details above and click "
        "'Analyse Property' to view the results."
    )

# =========================================================
# 9. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    🏠 <b>Real Estate Investment Advisor</b><br>
    Machine Learning and Data Analytics Project<br>
    Property Price Estimation | Investment Classification
</div>
""", unsafe_allow_html=True)