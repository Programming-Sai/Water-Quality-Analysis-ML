import streamlit as st
import joblib

# Load the model
model = joblib.load("./data/XGBoost_model.joblib")

st.set_page_config(
    page_title="Water Quality Analysis",  
    page_icon="./data/fav.png",
    menu_items={
        'Get Help': 'https://github.com/Programming-Sai/Water-Quality-Analysis-ML/issues',
        'Report a Bug': 'https://github.com/Programming-Sai/Water-Quality-Analysis-ML/issues',
        'About': 'This app predicts the quality of water based on various physicochemical, socio-economic, and environmental factors. It uses machine learning models to classify water samples as Clean or Dirty based on user input.'
    }
)

st.markdown(
    """
    <style>
    .stNumberInput { margin-bottom: 10px; } 
    .stMarkdown {margin-block: 30px}
    .stTooltipContent {
        background-color: #280330;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Water Quality Prediction")
st.markdown(
    """
    This tool predicts the quality of water based on several environmental and social factors.
    Provide the required inputs to determine if the water is **Clean** or **Dirty**.
    """
)

# Input features
feature_1 = st.number_input("Population Density (people/km²)", help=("Enter the population density of the region in people per square kilometer. ""This value impacts the level of human activities that could influence water quality."), value=None)
feature_2 = st.number_input("Waste Index", help=("Calculated as:\n"
        "(Maximum Waste Composition + Other Composition) ÷ Recycling Percentage.\n\n"
        "Waste compositions include:\n"
        "- Food Organic Waste\n"
        "- Glass\n"
        "- Plastic\n"
        "- Metal, etc.\n\n"
        "Higher values of the Waste Index signify poorer waste management, negatively affecting water quality."), value=None)
feature_3 = st.number_input("Development Index", help=("Calculated as:\nGDP × Literacy Rate (2010-2018).\n\n"
        "Higher values indicate better socio-economic development, which usually correlates with better water quality. "
        "Ensure that GDP is in trillions (e.g., 2.5 for $2.5 trillion)."), value=None)
# Add more inputs as needed...

# Predict button
if st.button("Predict"):
    features = [[feature_1, feature_2, feature_3]]  # Add all features here
    prediction = model.predict(features)
    result = 'Clean' if prediction[0] == 1 else 'Dirty'
    if result == 'Clean':
        st.success(f"Prediction: {result}")
    else:
        st.error(f"Prediction: {result}")


st.markdown("---")
st.caption("Powered by Machine Learning · Developed for demonstration purposes")

