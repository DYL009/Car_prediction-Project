import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load the cleaned car data
cleaned_car = pd.read_csv('cleaned_car.csv')

# Extract unique car models from the 'name' column
car_models = cleaned_car['name'].unique()

# Define the prediction function
def predict_price(car_model, company, year, driven, fuel_type):
    input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app title
st.title("Car Price Prediction Dashboard")

# Input fields
st.write("Please provide the following details to predict the car's price:")

# Use a selectbox populated with all unique car models from the dataset
car_model = st.selectbox("Car Model", car_models)

# Other input fields remain the same
company = st.selectbox("Company", cleaned_car['company'].unique())  # Dynamically populated
year = st.number_input("Year of Purchase", min_value=1990, max_value=2023, value=2019)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel_type = st.selectbox("Fuel Type", cleaned_car['fuel_type'].dropna().unique())  # Dynamically populated

# Predict button
if st.button("Predict Price"):
    # Predict the price using the input values
    price = predict_price(car_model, company, year, kms_driven, fuel_type)
    st.write(f"The estimated price of the car is ₹{int(price):,}")
