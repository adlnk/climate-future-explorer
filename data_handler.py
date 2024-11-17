# Standard library imports
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Third-party imports
import requests
import pandas as pd
import numpy as np
import anthropic
from dotenv import load_dotenv
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Initialize clients
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

# Basic climate metrics functions
def calculate_temp_mean(data, year):
    """Calculate mean temperature for a given year"""
    # Filter rows for current year and calculate mean of temperature_2m_mean
    temp_mean = (
        data[data['date'].dt.year == year]
        ['temperature_2m_mean']
        .mean()
    )
    return temp_mean

def calculate_seasonal_metrics(df, year):
    """Calculate seasonal statistics for a given year"""
    # Define seasons by months
    df = df[df['date'].dt.year == year]
    df['season'] = df['date'].dt.month.map({12:'winter', 1:'winter', 2:'winter',
                                          3:'spring', 4:'spring', 5:'spring',
                                          6:'summer', 7:'summer', 8:'summer',
                                          9:'fall', 10:'fall', 11:'fall'})
    
    seasonal_stats = df.groupby('season').agg({
        'temperature_2m_mean': ['mean', 'std'],
        'precipitation_sum': 'sum',
        'shortwave_radiation_sum': 'mean'
    })
    return seasonal_stats

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

def analyze_climate_data(df: pd.DataFrame, target_date: datetime, window_size: int = 5) -> Dict[str, Any]:
    """
    Analyze climate data with sophisticated temporal aggregation.
    
    Args:
        df: DataFrame with climate data
        target_date: Future date to analyze
        window_size: Size of window for aggregation (in years)
    """
    # Convert date to datetime if needed
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    def get_window_stats(center_date: datetime) -> Dict[str, Any]:
        """Calculate statistics for a window centered on a date"""
        center_year = center_date.year
        window_data = df[
            (df['year'] >= center_year - window_size//2) & 
            (df['year'] <= center_year + window_size//2)
        ]
        
        if len(window_data) == 0:
            return None

        # Average-based metrics
        means = {
            'temp_mean': window_data['temperature_2m_mean'].mean(),
            'cloud_cover': window_data['cloud_cover_mean'].mean(),
            'radiation': window_data['shortwave_radiation_sum'].mean(),
            'humidity_mean': (
                window_data['relative_humidity_2m_max'].mean() + 
                window_data['relative_humidity_2m_min'].mean()
            ) / 2
        }

        # Extreme metrics
        extremes = {
            'temp_max': window_data['temperature_2m_max'].max(),
            'temp_min': window_data['temperature_2m_min'].min(),
            'wind_max': window_data['wind_speed_10m_max'].max(),
            'humidity_max': window_data['relative_humidity_2m_max'].max(),
            'humidity_min': window_data['relative_humidity_2m_min'].min()
        }

        # Cumulative metrics
        annual_stats = window_data.groupby('year').agg({
            'precipitation_sum': 'sum',
            'snowfall_sum': 'sum'
        })
        
        cumulative = {
            'precip_annual_mean': annual_stats['precipitation_sum'].mean(),
            'precip_monthly_max': window_data['precipitation_sum'].max(),
            'snow_annual_mean': annual_stats['snowfall_sum'].mean(),
            'snow_monthly_max': window_data['snowfall_sum'].max()
        }

        # Extreme events
        temp_95th = window_data['temperature_2m_max'].quantile(0.95)
        precip_95th = window_data['precipitation_sum'].quantile(0.95)
        wind_95th = window_data['wind_speed_10m_max'].quantile(0.95)

        extreme_events = {
            'hot_days_annual': len(window_data[window_data['temperature_2m_max'] > temp_95th]) / window_size,
            'heavy_rain_annual': len(window_data[window_data['precipitation_sum'] > precip_95th]) / window_size,
            'high_wind_annual': len(window_data[window_data['wind_speed_10m_max'] > wind_95th]) / window_size
        }

        # Seasonal analysis
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11]
        }

        seasonal_stats = {}
        for season, months in seasons.items():
            season_data = window_data[window_data['month'].isin(months)]
            seasonal_stats[season] = {
                'temp_mean': season_data['temperature_2m_mean'].mean(),
                'temp_max': season_data['temperature_2m_max'].max(),
                'temp_min': season_data['temperature_2m_min'].min(),
                'precip_total': season_data['precipitation_sum'].mean() * 3,  # Approximate seasonal total
                'wind_max': season_data['wind_speed_10m_max'].max()
            }

        return {
            'means': means,
            'extremes': extremes,
            'cumulative': cumulative,
            'extreme_events': extreme_events,
            'seasonal': seasonal_stats
        }

    # Get current window (last 5 years of present data)
    current_data = get_window_stats(datetime.now())
    
    # Get future window (centered on target date)
    future_data = get_window_stats(target_date)
    
    # Calculate changes
    def compute_changes(current: Dict, future: Dict, prefix: str = '') -> Dict[str, float]:
        changes = {}
        for key in current.keys():
            if isinstance(current[key], dict):
                changes.update(compute_changes(current[key], future[key], f"{prefix}{key}_"))
            else:
                changes[f"{prefix}{key}_change"] = future[key] - current[key]
        return changes

    changes = compute_changes(current_data, future_data)

    return {
        'current': current_data,
        'future': future_data,
        'changes': changes
    }

def get_ai_analysis(location_name, df, year):
    """Get AI analysis of climate impact"""
    # Read the prompt template
    prompt_path = Path(__file__).parent / "prompts" / "climate_impact_prompt.txt"
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    # Get climate analysis data
    analysis_results = analyze_climate_data(df, year)
    
    # Prepare variables for template
    template_vars = {
        "LOCATION_NAME": location_name,
        "CURRENT_TEMP_MEAN": analysis_results['current']['means']['temp_mean'],
        "CURRENT_TEMP_MAX": analysis_results['current']['extremes']['temp_max'],
        "CURRENT_TEMP_MIN": analysis_results['current']['extremes']['temp_min'],
        "CURRENT_PRECIP_ANNUAL": analysis_results['current']['cumulative']['precip_annual_mean'],
        "CURRENT_PRECIP_MAX": analysis_results['current']['cumulative']['precip_monthly_max'],
        "FUTURE_TEMP_MEAN": analysis_results['future']['means']['temp_mean'],
        "FUTURE_TEMP_MAX": analysis_results['future']['extremes']['temp_max'],
        "FUTURE_TEMP_MIN": analysis_results['future']['extremes']['temp_min'],
        "FUTURE_PRECIP_ANNUAL": analysis_results['future']['cumulative']['precip_annual_mean'],
        "FUTURE_PRECIP_MAX": analysis_results['future']['cumulative']['precip_monthly_max'],
        "SEASONAL_CHANGES": str(analysis_results['changes']),
        "EXTREME_EVENTS": str(analysis_results['future']['extreme_events']),
        "VARIABILITY_METRICS": "Data not available",  # Could be added in future
        "TREND_METRICS": "Data not available"  # Could be added in future
    }
    
    # Fill the template
    filled_prompt = prompt_template.format(**template_vars)
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        temperature=0.4,
        system="You are a climate impact analyst. Provide detailed, evidence-based projections of climate change effects on daily life.",
        messages=[
            {"role": "user", "content": filled_prompt}
        ]
    )
    
    return message.content[0].text if isinstance(message.content, list) else message.content