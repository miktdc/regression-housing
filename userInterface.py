import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("California Housing Price Prediction")

# Create input fields for user data
latitude = st.number_input("Latitude", value=0.0, format="%.6f")
housing_median_age = st.number_input("Housing Median Age", value=0.0)
total_rooms = st.number_input("Total Rooms", value=0.0)
median_income = st.number_input("Median Income", value=0.0)
ocean_proximity_inland = st.radio("Is it Inland?", ["No", "Yes"]) == "Yes"
ocean_proximity_near_bay = st.radio("Is it Near Bay?", ["No", "Yes"]) == "Yes"
ocean_proximity_near_ocean = st.radio("Is it Near Ocean?", ["No", "Yes"]) == "Yes"

# Convert input to dataframe
input_data = pd.DataFrame(
    {
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "median_income": [median_income],
        "ocean_proximity_INLAND": [int(ocean_proximity_inland)],
        "ocean_proximity_NEAR BAY": [int(ocean_proximity_near_bay)],
        "ocean_proximity_NEAR OCEAN": [int(ocean_proximity_near_ocean)],
    }
)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Median House Value: ${prediction[0]:,.2f}")
