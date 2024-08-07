import dataiku
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle
from datetime import datetime as dt
from datetime import datetime
import os
import io
from PIL import Image
import base64

# Load the trained model from the pickle file

st.set_page_config(page_title="DET ONV - What-If", page_icon="https://cdn-icons-png.freepik.com/512/5632/5632376.png", layout="centered", initial_sidebar_state="auto", menu_items=None)


# Define the managed folder and the file path within the folder
folder = dataiku.Folder("artifacts2")
file_path = "random_forest_model.pkl"

# Download the file from the managed folder to the local environment
with folder.get_download_stream(file_path) as f:
    local_path = "/tmp/random_forest_model.pkl"
    with open(local_path, 'wb') as local_file:
        local_file.write(f.read())

# Load the pickle file from the local environment
with open(local_path, 'rb') as file:
    random_forest_model = pickle.load(file)
    
def load_model():
    folder = dataiku.Folder("artifacts2")
    file_path = "random_forest_model.pkl"
    
    # Download the file from the managed folder to the local environment
    with folder.get_download_stream(file_path) as f:
        local_path = "/tmp/random_forest_model.pkl"
        with open(local_path, 'wb') as local_file:
            local_file.write(f.read())
    
    # Load the pickle file from the local environment
    with open(local_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

# Define a function to convert image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Define a function to display the SHAP waterfall image based on the dropdown selections
def display_shap_waterfall(year, month, nationality):
    # Define the managed folder and the file path within the folder
    folder = dataiku.Folder("artifacts2")
    file_path = f"shap_plots/SHAP_Waterfall_{year}_{month}_{nationality}.png"

    # Download the file from the managed folder to the local environment
    with folder.get_download_stream(file_path) as f:
        image_data = f.read()

    # Display the image using Streamlit
    image = Image.open(io.BytesIO(image_data))
    st.image(image, caption=f"SHAP Waterfall for {year}-{month} ({nationality})")

# Streamlit App with tabs
st.title("Overnight Visitors Prediction")

# Define tabs
whatif_tab1, monthly_preds_tab2 = st.tabs(["What-If", "Monthly Predictions"])


with whatif_tab1:
    # Display the logo at the top of the sidebar
    st.sidebar.image("https://d2csxpduxe849s.cloudfront.net/media/AD103F47-75FF-486A-85096BE8028DB0BC/C27311A5-51B6-422C-B928E785925AC8EA/webimage-4F35A22B-AD2F-4F9A-8EA5F82D2D295A8C.png", width=250)
    
    # Pre-filled random values for demonstration
    st.sidebar.header("Essential Statics", divider='red')
    year = st.sidebar.selectbox('Year', list(range(dt.now().year, 2026)), index=1)
    
    month = st.sidebar.selectbox('Month', [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ], index=dt.now().month)
    num_months = st.sidebar.slider('Number of Months to Predict', min_value=1, max_value=12, value=3, key="num_months_slider")    
    
    nationality_string = st.sidebar.selectbox('Nationality', ['USA', 'UK', 'India', 'China', 'Germany'], index=1)

    # Encoding nationality as a numeric value
    nationality_dict = {'USA': 1, 'UK': 2, 'India': 3, 'China': 4, 'Germany': 5}
    nationality_encoded = nationality_dict[nationality_string]

    # Convert month to numeric value
    month_dict = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
        'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
        'November': 11, 'December': 12
    }
    month_numeric = month_dict[month]
    
    
    st.sidebar.header("What-If Scenario Creators", divider='red')
    # Input: Average ticket price for economy class
    economy_ticket_price = st.sidebar.number_input('Average Economy Ticket Price', min_value=0, value=500)
    
    # Input: Average ticket price for business class
    business_ticket_price = st.sidebar.number_input('Average Business Ticket Price', min_value=0, value=2000)

    avg_google_searches = st.sidebar.number_input('Average Google Searches In Previous Months', min_value=0, value=500)

    # Input: Number of bookings in last 3 months, last 2 months, last month
    bookings_last_3_months = st.sidebar.number_input('Number of Bookings in last 3 months', min_value=0, value=1000)
    bookings_last_2_months = st.sidebar.number_input('Number of Bookings in last 2 months', min_value=0, value=800)
    bookings_last_month = st.sidebar.number_input('Number of Bookings in last month', min_value=0, value=500)
    gdp = st.sidebar.number_input('GDP Value ($BN)', min_value=0, value=50)

    
    test_df = pd.DataFrame([year, month_numeric, nationality_encoded,economy_ticket_price, business_ticket_price, avg_google_searches, bookings_last_3_months, bookings_last_2_months, bookings_last_month, gdp]).T
    feature_name_list = ["Year", "Month", "Nationality", "Economy Price", "Business Price", "Avg Flight Searches", "Bookings Last 3 Months", "Bookings Last 2 Months", "Bookings Last Month", "G.D.P."]
    test_df.columns = feature_name_list
    test_df
    
    
    # Predict button
    if st.sidebar.button('Predict'):
        st.subheader("SHAP Waterfall Plot")

        for month_sequence in range(num_months):
            current_month = month_sequence+month_numeric
            month_name = next((k for k, v in month_dict.items() if v == current_month), None)
            if month_name != None:
                # st.write(f"For the Month of {month_name}")
                # st.write("Input Data")
                # test_df.values# Make predictions
                model = load_model()
                
                test_df['Month'] = month_dict[month_name]
                X_test = np.log(test_df.values + 1)  # Add 1 to avoid log(0)
                
                prediction_log = model.predict(X_test)                
                predicted_onv = np.exp(prediction_log)  # Reverse log transformation
        
                st.success(f"Predicted Number of Overnight Visitors from {nationality_string}, for {month_name} {year}: {int(predicted_onv[0])}")
                st.write(f"Prediction Input Data {month_name} {year}:\n")
                test_df
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_test)
                
                shap_values_obj = shap.Explanation(values=shap_values.values, base_values=shap_values.base_values, data=shap_values.data
                                                   , feature_names=feature_name_list
                                                  )
        
                # Display the waterfall plot for the first instance        
                fig, ax = plt.subplots()
                shap.waterfall_plot(shap_values_obj[0])
                st.pyplot(fig)
                st.header("", divider='red')
                st.header("\n\n")            
        
        
        
        
with monthly_preds_tab2:
    # Define the managed folder and the file path within the folder
    folder = dataiku.Folder("artifacts2")
    file_path = "onv_preds.csv"

    # Download the file from the managed folder to the local environment
    with folder.get_download_stream(file_path) as f:
        local_path = "/tmp/onv_preds.csv"
        with open(local_path, 'wb') as local_file:
            local_file.write(f.read())

    # Read the CSV file into a pandas DataFrame
    predictions_df = pd.read_csv(local_path)

    
    # Define CSS for image magnification
    st.markdown(
        """
        <style>
        .img-magnify {
            transition: transform 0.8s;
            width: 150px;
        }
        .img-magnify:hover {
            transform: scale(5);
            z-index: 1000;
            position: relative;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header("Upcoming Month's Predictions")

    # Model related metrics on top of the view
    # col1, col2, col3 = st.columns(3) 
    # col1.metric("Metric-1", "82/100", "12%")
    # col2.metric("Metric-2", "0.9/1.0", "-0.02")
    # col3.metric("Metric-3", "86/100", "4%")
    
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3], start=1):
        with col:
            st.subheader(f"Scenario {i}", divider="red")

            from_year = dt.now().year
            to_year = max(predictions_df['Year'])
            year_range = list(range(from_year, to_year + 1))

            year_index = max(0, min(len(year_range) - 1, i - 1))
            year = st.selectbox(f'Year', year_range, index=year_index, key=f'year{i}')

            month_index = (i - 1) % 12
            month = st.selectbox(f'Month',
                ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December'],
                index=month_index, key=f'month{i}').lower()

            nationality_index = (i - 1) % 5
            nationality = st.selectbox(f'Nationality', ['USA', 'UK', 'India', 'China', 'Germany'],
                index=nationality_index, key=f'nationality{i}').lower()

            filtered_df = predictions_df.loc[
                (predictions_df['Year'] == year) &
                (predictions_df['Month'] == month) &
                (predictions_df['Nationality'] == nationality)
            ]

            # Button to display data as per drow-down filters
            if st.button("Show Data", key=f'show_data_{i}'):
                st.write("Filtered DataFrame:")
                st.write(filtered_df)
            # Button to hide data
            st.button("Reset", type="primary", key=f'reset_{i}')

            # Determining predicted ONV from the excel file
            if not filtered_df.empty:
                visitor_count = filtered_df['ONV Predictions'].values[0]
            else:
                visitor_count = "Data not available"

            st.markdown(f"""
                <div>
                    <p>ONV Count: {visitor_count}</p>
                </div>
            """, unsafe_allow_html=True)
            # Display the SHAP waterfall image based on the selections
            display_shap_waterfall(year, month, nationality)
            st.write(f"{nationality.upper()} Visitors for {month.capitalize()} - {year}");
    # 3 column layout ends here
    
    # Add a section at the bottom for the line chart
    st.subheader(" ")
    st.subheader(" ", divider="red")
    st.subheader(" ")
    
    st.header("ONV Count For Upcoming Months")
    
    # Create a 'Year-Month' column in predictions_df
    if 'Year-Month' not in predictions_df.columns:
        predictions_df['Year-Month'] = predictions_df['Year'].astype(str) + '-' + predictions_df['Month'].str.capitalize()
    
    # Define available years as current year and next year
    current_year = dt.now().year
    available_years = [current_year, current_year + 1]
    
    # Get unique values for selection
    available_months = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
    available_nationalities = predictions_df['Nationality'].unique()
    
    # Calculate the next three months
    current_date = datetime.now()
    next_three_months = [(current_date.month + i - 1) % 12 + 1 for i in range(1, 4)]
    next_three_months_names = [current_date.replace(month=month).strftime('%B') for month in next_three_months]
    
    # Sidebar selections for Year, Month, and Nationality
    selected_year = st.selectbox("Select Year", available_years, index=0, key='line_chart_year')
    selected_months = st.multiselect("Select Month", available_months, default=next_three_months_names, key='line_chart_months')
    selected_months_lower = [month.lower() for month in selected_months]
    selected_nationalities = st.multiselect("Select Nationality", available_nationalities, default=available_nationalities[0], key='line_chart_nationalities')
    selected_nationalities_lower = [nat.lower() for nat in selected_nationalities]
    
    # Filter predictions_df based on selections
    filtered_df_for_chart = predictions_df[(predictions_df['Year'] == selected_year) &
                                           (predictions_df['Month'].isin(selected_months_lower)) &
                                           (predictions_df['Nationality'].isin(selected_nationalities_lower))]
    
    if not filtered_df_for_chart.empty:
        # Create line chart
        fig, ax = plt.subplots()
        for nationality in selected_nationalities_lower:
            nat_df = filtered_df_for_chart[filtered_df_for_chart['Nationality'] == nationality]
            ax.plot(nat_df['Year-Month'], nat_df['ONV Predictions'], marker='o', label=nationality.capitalize())
        
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Visitor Count")
        selected_months_str = ', '.join([month.capitalize() for month in selected_months])
        ax.set_title(f"ONV Count For Upcoming Months in {selected_year}")
        ax.legend(title="Nationality")
        
        st.pyplot(fig)
    else:
        st.write("No data available for the selected criteria.")