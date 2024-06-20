import streamlit as st
import numpy as np

# Sample model for prediction
def predict_visitors(inputs):
    # Dummy regression model coefficients
    coef = np.array([50, 200, 300, 0.8, 1.2, 2.1, -0.5, 1.1, 0.7, -150, 0.5, 100, 200])
    intercept = 10000
    return np.dot(inputs, coef) + intercept

# Streamlit App
st.title("Overnight Visitors Prediction")

# Input: Year, Month, Nationality
year = st.selectbox('Year', list(range(2020, 2025)))
month = st.selectbox('Month', list(range(1, 13)))
nationality = st.selectbox('Nationality', ['USA', 'UK', 'India', 'China', 'Germany'])

# Encoding nationality as a numeric value
nationality_dict = {'USA': 1, 'UK': 2, 'India': 3, 'China': 4, 'Germany': 5}
nationality_encoded = nationality_dict[nationality]

# Input: Economy Ticket Price, Business Ticket Price
economy_ticket_price = st.number_input('Economy Ticket Price', min_value=0)
business_ticket_price = st.number_input('Business Ticket Price', min_value=0)

# Input: Number of bookings in last 3 months, last 2 months, last month
bookings_last_3_months = st.number_input('Number of bookings in last 3 months', min_value=0)
bookings_last_2_months = st.number_input('Number of bookings in last 2 months', min_value=0)
bookings_last_month = st.number_input('Number of bookings in last month', min_value=0)

# Input: Number of Searches in last 3 months, last 2 months, last month
searches_last_3_months = st.number_input('Number of Searches in last 3 months', min_value=0)
searches_last_2_months = st.number_input('Number of Searches in last 2 months', min_value=0)
searches_last_month = st.number_input('Number of Searches in last month', min_value=0)

# Input: Unemployment rate
unemployment_rate = st.number_input('Unemployment rate', min_value=0.0, max_value=100.0, format="%.2f")

# Input: Marketing spend in last 12 months
marketing_spend = st.number_input('Marketing spend in last 12 months', min_value=0)

# Button to trigger prediction
if st.button('Predict Number of Overnight Visitors'):
    # Prepare input data for prediction
    inputs = np.array([
        year, month, nationality_encoded, economy_ticket_price, business_ticket_price, 
        bookings_last_3_months, bookings_last_2_months, bookings_last_month, 
        searches_last_3_months, searches_last_2_months, searches_last_month, 
        unemployment_rate, marketing_spend
    ])

    # Prediction
    prediction = predict_visitors(inputs)
    st.write(f"Predicted Number of Overnight Visitors: {int(prediction)}")
