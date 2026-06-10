import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("bangalore_xgboost_model.pkl")
locations = joblib.load("locations.pkl")

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f77b4;
}

.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #f0f8ff;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown(
    '<p class="main-title">🏠 Bangalore House Price Predictor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Predict property prices using Machine Learning & XGBoost</p>',
    unsafe_allow_html=True
)

# ==========================
# Sidebar Inputs
# ==========================
st.sidebar.header("Property Details")

area_type = st.sidebar.selectbox(
    "Area Type",
    [
        "Super built-up Area",
        "Built-up Area",
        "Plot Area",
        "Carpet Area"
    ]
)

location = st.sidebar.selectbox(
    "Location",
    locations
)

availability = st.sidebar.selectbox(
    "Availability",
    [
        "Ready To Move",
        "Immediate Possession"
    ]
)

total_sqft = st.sidebar.number_input(
    "Total Sqft",
    min_value=300.0,
    value=1200.0
)

bhk = st.sidebar.slider(
    "BHK",
    min_value=1,
    max_value=10,
    value=2
)

bath = st.sidebar.slider(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

balcony = st.sidebar.slider(
    "Balconies",
    min_value=0,
    max_value=5,
    value=1
)

# ==========================
# Property Summary
# ==========================
st.subheader("📋 Property Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("BHK", bhk)
col2.metric("Bathrooms", bath)
col3.metric("Balconies", balcony)
col4.metric("Area", f"{int(total_sqft)} sqft")

# ==========================
# Prediction
# ==========================
if st.button("🔮 Predict Price", use_container_width=True):

    sample = pd.DataFrame({
        "area_type": [area_type],
        "availability": [availability],
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "balcony": [balcony],
        "bhk": [bhk]
    })

    prediction = model.predict(sample)[0]

    st.markdown(
        f"""
        <div class="prediction-box">
            Estimated Property Price <br>
            ₹ {prediction:,.2f} Lakhs
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

# ==========================
# Footer
# ==========================
st.markdown(
    """
    <div class="footer">
        Built with Streamlit • XGBoost • Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
