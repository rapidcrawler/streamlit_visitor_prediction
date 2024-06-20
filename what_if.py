# Includes 
#  - log-antilog
#  - pickle file refrence
#  - SHAP Plot
#  - logo in sidebar


import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle

# Load the trained model from the pickle file
artifact_path = './artifacts/'
with open(artifact_path+'linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.title("Overnight Visitors Prediction")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Display the logo at the top of the sidebar
st.sidebar.image("https://uaerg.ae/wp-content/uploads/2022/10/det-report-logos.png", width=250)

# Pre-filled random values for demonstration
year = st.sidebar.selectbox('Year', list(range(2020, 2025)), index=2)
month = st.sidebar.selectbox('Month', [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], index=5)
nationality = st.sidebar.selectbox('Nationality', ['USA', 'UK', 'India', 'China', 'Germany'], index=1)

# Encoding nationality as a numeric value
nationality_dict = {'USA': 1, 'UK': 2, 'India': 3, 'China': 4, 'Germany': 5}
nationality_encoded = nationality_dict[nationality]

# Input: Economy Ticket Price, Business Ticket Price
economy_ticket_price = st.sidebar.number_input('Economy Ticket Price', min_value=0, value=300)
business_ticket_price = st.sidebar.number_input('Business Ticket Price', min_value=0, value=800)

# Input: Number of bookings in last 3 months, last 2 months, last month
bookings_last_3_months = st.sidebar.number_input('Number of bookings in last 3 months', min_value=0, value=500)
bookings_last_2_months = st.sidebar.number_input('Number of bookings in last 2 months', min_value=0, value=400)
bookings_last_month = st.sidebar.number_input('Number of bookings in last month', min_value=0, value=300)

# Input: Number of Searches in last 3 months, last 2 months, last month
searches_last_3_months = st.sidebar.number_input('Number of Searches in last 3 months', min_value=0, value=1500)
searches_last_2_months = st.sidebar.number_input('Number of Searches in last 2 months', min_value=0, value=1200)
searches_last_month = st.sidebar.number_input('Number of Searches in last month', min_value=0, value=1000)

# Input: Unemployment rate
unemployment_rate = st.sidebar.number_input('Unemployment rate', min_value=0.0, max_value=100.0, format="%.2f", value=5.0)

# Input: Marketing spend in last 12 months
marketing_spend = st.sidebar.number_input('Marketing spend in last 12 months', min_value=0, value=10000)

# Button to trigger prediction
predict_button = st.sidebar.button('Predict Number of Overnight Visitors')

if predict_button:
    # Prepare input data for prediction
    inputs = np.array([
        year, month.index(month) + 1, nationality_encoded, economy_ticket_price, business_ticket_price, 
        bookings_last_3_months, bookings_last_2_months, bookings_last_month, 
        searches_last_3_months, searches_last_2_months, searches_last_month, 
        unemployment_rate, marketing_spend
    ]).reshape(1, -1)

    # Apply log transformation
    inputs_log = np.log(inputs + 1)  # Add 1 to avoid log(0)
    
    # Prediction with a loading animation
    with st.spinner('Predicting...'):
        prediction_log = model.predict(inputs_log)
        prediction = np.exp(prediction_log)  # Reverse log transformation
    
    # Display the prediction result with a subtle success message
    st.success(f"Predicted Number of Overnight Visitors from {nationality}, for {month} {year}: {int(prediction[0])}")

    # Explain the model's predictions using SHAP
    explainer = shap.Explainer(model, np.log(np.zeros((1, 13)) + 1))
    shap_values = explainer(inputs_log)

    st.subheader("SHAP Waterfall Plot")
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0])
    st.pyplot(plt)
