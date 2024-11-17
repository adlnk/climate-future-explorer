import requests
import pandas as pd
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def get_location_data(address):
    """Get latitude, longitude and location name from address"""
    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={address}&count=1"
    response = requests.get(geocoding_url)
    data = response.json()
    
    if data.get("results"):
        return (
            data["results"][0]["latitude"],
            data["results"][0]["longitude"],
            data["results"][0]["name"]
        )
    return None

def get_climate_data(lat, lon):
    """Fetch and process climate projection data"""
    climate_url = f"https://climate-api.open-meteo.com/v1/climate?latitude={lat}&longitude={lon}&models=EC_Earth3P_HR&daily=temperature_2m_max,precipitation_sum&start_date=2020-01-01&end_date=2050-12-31"
    climate_data = requests.get(climate_url).json()
    
    return pd.DataFrame({
        'date': pd.to_datetime(climate_data['daily']['time']),
        'temp_max': climate_data['daily']['temperature_2m_max'],
        'precipitation': climate_data['daily']['precipitation_sum']
    })

def get_ai_analysis(location_name, df, financial_interest, demographic_interest, climate_interest, year):
    """Get AI analysis of climate impact"""
    prompt = f"""I live in {location_name}. In the year {year}, how will my day-to-day life be different from how it was in the year 2024? 

Please provide your response in two distinct parts:

PART 1: Write a vivid, first-person narrative (about 200 words) describing a typical day in my life 5 years from now, focusing on climate change impacts. Make it personal and emotionally resonant.

PART 2: Provide 5-7 concrete statistics and projections that support the narrative above.

In your response, please emphasize:
{"- Financial impacts (housing, utilities, insurance, adaptation costs)" if financial_interest else ""}
{"- Demographic changes and community effects" if demographic_interest else ""}
{"- Local climate changes and their direct effects" if climate_interest else ""}

Base your response on climate projections showing:
- Average maximum temperature: {df['temp_max'].mean():.1f}°C
- Average precipitation: {df['precipitation'].mean():.1f}mm
"""
    
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=0.4,
        system="You are a climate impact analyst. Provide detailed, evidence-based projections of climate change effects on daily life.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text if isinstance(message.content, list) else message.content
