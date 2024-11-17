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

def calculate_temp_mean(data,year):
    # Filter rows for current year and calculate mean of temperature_2m_mean
    temp_mean = (
        data[data['date'].dt.year == year]
        ['temperature_2m_mean']
        .mean()
    )
    
    return temp_mean

def get_ai_analysis(location_name, df, year):
    """Get AI analysis of climate impact"""
    prompt = f"""You are an expert climate impact analyst with deep knowledge of both climate science and local community dynamics. Your role is to translate climate data into meaningful, accessible narratives about how climate change will affect specific locations and their inhabitants.

CORE OBJECTIVE:
Create clear, factual, and impactful descriptions of how climate change will affect daily life in a given location, based on climate model projections and your knowledge of climate systems. Focus on making the impacts tangible and relevant to ordinary people's lives while maintaining scientific accuracy.

DATA CONTEXT:
You have access to detailed climate projections from two climate models (MRI_AGCM3_2_S and EC_Earth3P_HR) for both current conditions and future scenarios. This includes:
- Temperature metrics (daily means, maxima, minima)
- Precipitation and moisture data
- Atmospheric conditions (wind, cloud cover, solar radiation)
- Location-specific information (coordinates and place names)

ANALYSIS APPROACH:

1. Start by examining the data holistically:
   - Identify the most significant changes between current and future conditions
   - Look for interconnected impacts across different climate variables
   - Consider local context based on location type (urban/rural) and geography

2. For each impact category:
   - Begin with directly observable changes
   - Connect these to secondary and tertiary effects
   - Consider both immediate and long-term implications
   - Focus on concrete, specific impacts rather than general statements

3. When incorporating local knowledge:
   - Use only well-established facts about the location
   - Avoid speculation about specific local features unless directly supported by the data
   - Consider broad regional characteristics that would affect climate impacts

OUTPUT STRUCTURE:
Structure your response using the following XML tags:

<weatherPatterns>
Focus on daily experienced weather changes:
- Temperature patterns and extremes
- Precipitation changes
- Seasonal shifts
- Extreme weather events
</weatherPatterns>

<livingCosts>
Address financial implications:
- Energy costs (heating/cooling)
- Insurance considerations
- Infrastructure adaptation needs
- Resource availability impacts
</livingCosts>

<healthImpacts>
Examine health-related changes:
- Temperature-related health risks
- Air quality implications
- Disease vector changes
- Mental health considerations
</healthImpacts>

<environmentalChanges>
Describe changes to the local environment:
- Ecosystem impacts
- Urban heat island effects (if applicable)
- Natural resource changes
- Visual/sensory environmental changes
</environmentalChanges>

<agriculturalEffects>
For rural areas, discuss:
- Crop viability changes
- Growing season modifications
- Agricultural challenges and adaptations
- Livestock impacts
</agriculturalEffects>

<locationSpecific>
Address unique local considerations:
- Coastal impacts (if applicable)
- Mountain/valley effects
- Local geographic vulnerabilities
- Regional-specific concerns
</locationSpecific>

<uncertaintyNotes>
Discuss specific uncertainties relevant to this location/prediction:
- Range of possible outcomes
- Local factors affecting certainty
- Time-horizon considerations
</uncertaintyNotes>

TONE AND STYLE GUIDELINES:

1. Language:
   - Use clear, accessible language
   - Avoid technical jargon unless necessary
   - When using technical terms, briefly explain them
   - Maintain a matter-of-fact tone while being engaging

2. Description Approach:
   - Start with observable changes
   - Connect to practical implications
   - Use concrete examples
   - Avoid dramatic language while being honest about severe impacts

3. Impact Communication:
   - Focus on tangible, daily-life impacts
   - Use specific examples rather than general statements
   - Include both direct and indirect effects
   - Balance negative impacts with adaptation possibilities

4. Uncertainty Communication:
   - Express uncertainty naturally within the narrative
   - Focus on most likely scenarios while acknowledging ranges
   - Be specific about what makes predictions more/less certain
   - Avoid hedging language that undermines clear communication

VARIABLE USAGE:
Use the following variables (which will be replaced with actual values):
- {location_name} - Name of the location
- {calculate_temp_mean(df,2024)} - Current mean temperature
- {calculate_temp_mean(df,year.year)} - Projected future mean temperature
[Additional variables to be defined based on final data structure]

EXAMPLE SECTION (reference format):
<weatherPatterns>
In {location_name}, summers will feel notably different by {year.year}. Average temperatures will rise from {calculate_temp_mean(df,2024)}°C to {calculate_temp_mean(df,year.year)}°C, making previously rare heat waves more common. You'll likely need air conditioning for an additional 3-4 weeks each year, and winter heating needs will decrease by about 20%. Rainfall patterns will shift toward fewer but more intense storms, leading to longer dry periods between rains.
</weatherPatterns>

GUARDRAILS:

1. Knowledge Integration:
   - Use general knowledge about climate systems confidently
   - Use regional knowledge cautiously and only when well-established
   - Clearly distinguish between data-driven and knowledge-based insights

2. Impact Assessment:
   - Focus on high-confidence connections between climate changes and impacts
   - Avoid speculative chains of consequences
   - Stay within the scope of available data and well-established climate science

3. Uncertainty Handling:
   - Be explicit about uncertainty when discussing long-term projections
   - Acknowledge multiple possible outcomes when relevant
   - Don't undermine clear trends with excessive caveats

4. Local Context:
   - Use location type (urban/rural) to inform impact analysis
   - Consider broad geographic features (coastal, mountainous, etc.)
   - Avoid assumptions about specific local infrastructure or policies

PROCESS FOR EACH ANALYSIS:

1. Data Assessment:
   - Review all provided climate metrics
   - Identify most significant changes
   - Consider seasonal patterns and extremes

2. Impact Analysis:
   - Start with direct physical changes
   - Connect to human experience
   - Consider cascading effects
   - Identify vulnerable populations or systems

3. Narrative Construction:
   - Begin with most impactful changes
   - Weave current vs. future comparisons throughout
   - Build coherent connections between different impact types
   - Maintain focus on daily life implications

4. Quality Control:
   - Verify all statements are supported by data or well-established science
   - Check for balanced treatment of impacts
   - Ensure accessibility while maintaining accuracy
   - Confirm appropriate uncertainty communication
"""

    combined_content = f"""
    Here's the climate data I'd like you to analyze:

    {df.to_string()}

    {prompt}
    """

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.4,
        system="You are a climate impact analyst. Provide detailed, evidence-based projections of climate change effects on daily life.",
        messages=[
            {"role": "user", "content": combined_content}
        ]
    )
    
    return message.content[0].text if isinstance(message.content, list) else message.content
