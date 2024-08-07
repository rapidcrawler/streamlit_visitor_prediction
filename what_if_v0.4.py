import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle

# Load the trained model from the pickle file
artifact_path = './artifacts/'
with open(artifact_path + 'linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App with tabs
st.title("Overnight Visitors Prediction")

# Define tabs
tab1, tab2 = st.tabs(["What-If", "Monthly Predictions"])

with tab1:
    # Sidebar for inputs

    # Display the logo at the top of the sidebar
    st.sidebar.image("https://uaerg.ae/wp-content/uploads/2022/10/det-report-logos.png", width=250)

    st.sidebar.header("DET - Overnight Visitors Prediction")

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

    # Convert month to numeric value
    month_dict = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
        'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
        'November': 11, 'December': 12
    }
    month_numeric = month_dict[month]

    # Input: Average ticket price for economy class
    economy_ticket_price = st.sidebar.number_input('Average Economy Ticket Price', min_value=0, value=500)

    # Input: Average ticket price for business class
    business_ticket_price = st.sidebar.number_input('Average Business Ticket Price', min_value=0, value=2000)

    # Input: Number of bookings in last 3 months, last 2 months, last month
    bookings_last_3_months = st.sidebar.number_input('Number of Bookings in last 3 months', min_value=0, value=1000)
    bookings_last_2_months = st.sidebar.number_input('Number of Bookings in last 2 months', min_value=0, value=800)
    bookings_last_month = st.sidebar.number_input('Number of Bookings in last month', min_value=0, value=500)

    # Predict button
    if st.sidebar.button('Predict'):
        # Combine inputs into a single array
        inputs = np.array([year, month_numeric, nationality_encoded, economy_ticket_price, business_ticket_price,
                   bookings_last_3_months, bookings_last_2_months, bookings_last_month, 0, 0, 0, 0, 0], dtype=float)
        inputs_log = np.log(inputs + 1)  # Add 1 to avoid log(0)

        # Perform the prediction
        prediction_log = model.predict(inputs_log.reshape(1, -1))
        prediction = np.exp(prediction_log)  # Reverse log transformation

        # Display the prediction result with a subtle success message
        st.success(f"Predicted Number of Overnight Visitors from {nationality}, for {month} {year}: {int(prediction[0])}")

        # Explain the model's predictions using SHAP
        explainer = shap.Explainer(model, np.log(np.zeros((1, 13)) + 1))

        shap_values = explainer(inputs_log.reshape(1, -1))

        st.subheader("SHAP Waterfall Plot")
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0])
        st.pyplot(plt)

with tab2:
    # Create three columns for the sub-sections
    col1, col2, col3 = st.columns(3)

    for i, col in enumerate([col1, col2, col3], start=1):
        with col:
            st.subheader(f"Sub-section {i}")

            # Dropdowns for Year, Month, and Nationality
            year = st.selectbox(f'Year', list(range(2020, 2025)), index=i-1, key=f'year{i}')
            month = st.selectbox(f'Month', [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ], index=(i-1) % 12, key=f'month{i}')
            nationality = st.selectbox(f'Nationality', ['USA', 'UK', 'India', 'China', 'Germany'], index=(i-1) % 5, key=f'nationality{i}')

            # Simulate visitor count (replace with actual calculation if available)
            visitor_count = np.random.randint(5000, 20000)  # Random number for demonstration purposes

            # Display the visitor count in a highlighted cell box
            st.markdown(f"""
                <div>
                    <p>ONV Count: {visitor_count}</p>
                </div>
            """, unsafe_allow_html=True)
