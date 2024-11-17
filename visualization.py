import plotly.express as px

def create_temperature_plot(df):
    """Create temperature trend visualization"""
    fig = px.line(df, x='date', y='temp_max',
                  title='Maximum Temperature Projection',
                  labels={'date': 'Year', 'temp_max': 'Temperature (°C)'},
                  line_shape='spline')
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

def create_precipitation_plot(df):
    """Create precipitation trend visualization"""
    fig = px.line(df, x='date', y='precipitation',
                  title='Precipitation Projection',
                  labels={'date': 'Year', 'precipitation': 'Precipitation (mm)'},
                  line_shape='spline')
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified'
    )
    return fig
