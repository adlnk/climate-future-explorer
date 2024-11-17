import streamlit as st
import datetime
from data_handler import get_location_data, get_climate_data, get_ai_analysis
from visualization import create_temperature_plot, create_precipitation_plot
from bs4 import BeautifulSoup
import pandas as pd
import re

st.set_page_config(page_title="Climate Future Explorer", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 16px;
        font-size: 14px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Climate Future Explorer")
st.write("""
Explore how climate change could affect your daily life in the next 5-10 years. 
Enter your location and adjust the sliders to focus on aspects that matter most to you. 
We'll provide a personalized analysis of how your life might change, backed by data and research.
""")

col1, col2 = st.columns([2, 1])

with col1:
    address = st.text_input("Enter your address:", placeholder="e.g., 1234 Main St, Seattle, WA 98101")

    submit = st.button("Analyze Future Impact", type="primary")

with col2:
    year = st.date_input('Enter your date of interest.',
                         min_value = datetime.date(2029, 1, 1),
                         max_value=datetime.date(2050,1,1),
                         format = 'DD/MM/YYYY')

def extract_section_content(response_text, tag_name):
    """Extract content between XML-like tags"""
    try:
        # Make the pattern more flexible to handle potential whitespace and newlines
        pattern = f"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            return content
        return ""
    except Exception as e:
        st.error(f"Error extracting {tag_name}: {str(e)}")
        return ""

if submit and address:
    try:
        # Get location data
        location_data = get_location_data(address)
        if location_data:
            lat, lon, location_name = location_data
            
            # Get climate data
            df = get_climate_data(lat, lon)

            # Get AI analysis
            response_text = get_ai_analysis(location_name, df, year)
            
            # Debug: Print raw response
            st.write("Debug - Raw Response:", response_text[:200] + "...")
            
            # Display results
            st.header(f"What {location_name} will look like in the year {year.year}?")
            
            # Weather Patterns Section
            with st.expander("🌡️ Weather Pattern Changes", expanded=True):
                weather_content = extract_section_content(response_text, "weatherPatterns")
                # Debug: Print extracted content
                st.write("Debug - Extracted Weather Content:", weather_content[:200] if weather_content else "No content found")
                if weather_content:
                    st.markdown(weather_content)
                else:
                    st.warning("No weather pattern data found in the response")

            # Health Impacts Section
            with st.expander("🤒 How different you will feel physically", expanded=True):
                health_content = extract_section_content(response_text, "healthImpacts")
                st.markdown(health_content)

            # Financial Impact Section
            with st.expander("💰 How will the climate affect my wallet?", expanded=True):
                costs_content = extract_section_content(response_text, "livingCosts")
                st.markdown(costs_content)

            # Environmental Changes Section
            with st.expander("🌳 Environmental Impact", expanded=True):
                env_content = extract_section_content(response_text, "environmentalChanges")
                st.markdown(env_content)

            # Agricultural Effects Section
            with st.expander("🌾 Agricultural Changes", expanded=True):
                agri_content = extract_section_content(response_text, "agriculturalEffects")
                st.markdown(agri_content)

            # Comfort Analysis Section
            with st.expander("🌡️ Comfort & Living Conditions", expanded=True):
                comfort_content = extract_section_content(response_text, "comfort_analysis")
                st.markdown(comfort_content)

            # Energy Implications Section
            with st.expander("⚡ Energy Impact", expanded=True):
                energy_content = extract_section_content(response_text, "energy_implications")
                st.markdown(energy_content)

            # Detailed Seasonal Changes
            with st.expander("🗓️ Detailed Seasonal Changes", expanded=True):
                seasonal_content = extract_section_content(response_text, "seasonal_details")
                st.markdown(seasonal_content)

            # Outdoor Activities Impact
            with st.expander("🏃‍♂️ Outdoor Activities Impact", expanded=True):
                outdoor_content = extract_section_content(response_text, "outdoor_activities")
                st.markdown(outdoor_content)

            # Uncertainty Notes Section
            with st.expander("ℹ️ Uncertainty Factors", expanded=True):
                uncertainty_content = extract_section_content(response_text, "uncertaintyNotes")
                st.markdown(uncertainty_content)

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