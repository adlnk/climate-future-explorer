import streamlit as st
import datetime
from data_handler import get_location_data, get_climate_data, get_ai_analysis
from visualization import create_temperature_plot, create_precipitation_plot

# Page configuration
st.set_page_config(page_title="Climate Future Explorer", layout="wide")

# App header and description
st.title("Climate Future Explorer")
st.write("""
Explore how climate change could affect your daily life in the next 5-10 years. 
Enter your location and adjust the sliders to focus on aspects that matter most to you. 
We'll provide a personalized analysis of how your life might change, backed by data and research.
""")

# Create two columns for input
col1, col2 = st.columns([2, 1])

with col1:
    # Location input
    address = st.text_input("Enter your address:", placeholder="e.g., 1234 Main St, Seattle, WA 98101")

    # Submit button
    submit = st.button("Analyze Future Impact", type="primary")

with col2:
    year = st.date_input('Enter your date of interest.',
                         min_value = datetime.date(2029, 1, 1),
                         max_value=datetime.date(2050,1,1),
                         format = 'DD/MM/YYYY')



# Only process if submit is clicked and address is provided
if submit and address:
    try:
        # Get location data
        location_data = get_location_data(address)
        if location_data:
            lat, lon, location_name = location_data
            
            # Get climate data
            df = get_climate_data(lat, lon)
            st.dataframe(df)

            # Get AI analysis
            response_text = get_ai_analysis(location_name, df, year)
            
            # Display results
            st.header("Your Climate Future")
            
            st.write(response_text)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("""
    This tool combines climate science, demographic data, and economic projections 
    to help you understand how climate change might affect your daily life.
    """)
    
    st.header("Data Sources")
    st.write("""
    - Climate projections: Open-Meteo Climate API
    - Demographic data: Various government sources
    - Economic projections: World Bank Climate Change Knowledge Portal
    """)
    
    st.header("Disclaimer")
    st.write("""
    Projections are based on current climate models and trends. 
    Actual future conditions may vary due to policy changes, technological advances, 
    and other factors.
    """)
