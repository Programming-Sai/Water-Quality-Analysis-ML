import streamlit as st
import joblib

# Load the model
model = joblib.load("data/XGBoost_model.joblib")

st.markdown(
    """
    <style>
    .stNumberInput { margin-bottom: 10px; } 
    .stMarkdown {margin-block: 30px}
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
feature_1 = st.number_input("Population Density", help=("Enter the population density of the region in people per square kilometer. ""This value impacts the level of human activities that could influence water quality."))
feature_2 = st.number_input("Waste Index", help=("The Waste Index is calculated based on the percentage of waste materials in the area, " "scaled between 0 and 100. For example:\n\n" "- A region with 30% plastic and 20% organic waste has a Waste Index of 50.\n" "- Higher values indicate more waste, which negatively impacts water quality."))
feature_3 = st.number_input("Development Index", help=("The Development Index measures the socio-economic development of the region, scaled between 0 and 100. " "It considers factors like:\n\n" "- Access to clean water and sanitation\n" "- Infrastructure development\n" "- Education and literacy rates\n\n" "Higher values indicate better development, which generally correlates with improved water quality."))
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



