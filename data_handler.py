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

def get_climate_data(lat, lon, start_date="1950-01-01", end_date="2050-12-31"):
    """Fetch comprehensive climate data and aggregate to monthly"""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    # Setup the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "models": ["MRI_AGCM3_2_S", "EC_Earth3P_HR"],
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
            "cloud_cover_mean",
            "shortwave_radiation_sum",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "precipitation_sum",
            "snowfall_sum"
        ]
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process daily data
    daily = response.Daily()
    
    # Create daily DataFrame
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
        "wind_speed_10m_max": daily.Variables(3).ValuesAsNumpy(),
        "cloud_cover_mean": daily.Variables(4).ValuesAsNumpy(),
        "shortwave_radiation_sum": daily.Variables(5).ValuesAsNumpy(),
        "relative_humidity_2m_max": daily.Variables(6).ValuesAsNumpy(),
        "relative_humidity_2m_min": daily.Variables(7).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(8).ValuesAsNumpy(),
        "snowfall_sum": daily.Variables(9).ValuesAsNumpy()
    }

    daily_df = pd.DataFrame(data=daily_data)

    # Aggregate to monthly
    monthly_df = daily_df.set_index('date').resample('M').agg({
        'temperature_2m_mean': 'mean',
        'temperature_2m_max': 'mean',
        'temperature_2m_min': 'mean',
        'wind_speed_10m_max': 'mean',
        'cloud_cover_mean': 'mean',
        'shortwave_radiation_sum': 'sum',
        'relative_humidity_2m_max': 'mean',
        'relative_humidity_2m_min': 'mean',
        'precipitation_sum': 'sum',
        'snowfall_sum': 'sum'
    }).reset_index()

    return monthly_df



def get_ai_analysis(location_name, df, year):
    """Get AI analysis of climate impact"""
    prompt = f"""You are an expert climate analyst tasked with creating a personal, first-person narrative about life in [LOCATION] in the year [YEAR]. Using web searches for current climate data such as this data:{df}, scientific projections, and local planning documents, paint a vivid picture of how climate change has transformed daily life. Your narrative should feel authentic while being grounded in scientific evidence.
Begin with: 'Let me take you through a typical day in {location_name} in {year.year}...'
As you craft the narrative, weave in detailed analysis of:
	1.	Daily Life & Weather
	•	How typical weather patterns affect daily routines
	•	Changes in seasonal activities
	•	Adaptations to new climate norms
Include specific temperature changes, precipitation patterns, and compare to historical baselines
	2.	Environmental Realities
	•	Visible changes to local landscapes and ecosystems
	•	New weather phenomena residents face
	•	Changes to local flora and fauna
Support with projected environmental data and research
	3.	Community Impact
	•	How neighborhoods and communities have adapted
	•	Changes to local economy and jobs
	•	Population shifts and demographic changes
	•	New infrastructure and adaptation measures
Reference relevant economic and demographic projections
	4.	Personal Adaptations
	•	How homes and daily routines have changed
	•	New technologies and practices for climate resilience
	•	Community support systems and networks
Include current adaptation plans and projected needs
Throughout the narrative, include relevant links to scientific sources, climate models, and local planning documents that support your projections. Balance engaging storytelling with factual accuracy, using real data to ground the narrative.
End with a reflection on how this future compares to present day conditions, highlighting both challenges and adaptations that have emerged.
Remember to perform web searches to validate projections and include the most current climate science in your analysis.
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
