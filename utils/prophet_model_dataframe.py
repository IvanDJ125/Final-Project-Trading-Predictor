import os
import sys
import pandas as pd
import logging
import warnings
from prophet import Prophet
from utils.stock_data import get_stock_data
import plotly.graph_objects as go
# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific warnings related to missing backends in transformers
warnings.filterwarnings("ignore", category=UserWarning, message="None of PyTorch, TensorFlow.*")

# Suppress cmdstanpy informational logs
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

# Suppress warnings from the transformers library
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# Suppress info logs from cmdstanpy
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# Add the project root directory to sys.path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

def predict_and_plot_prophet(ticker, forecast_period=30):
    """
    Use the Prophet model to predict stock prices and plot the results.

    Args:
        ticker (str): Stock ticker symbol.
        forecast_period (int): Number of days to forecast (default is 30).

    Returns:
        dict: A summary of the forecast, including the latest predicted price.
        plotly.graph_objects.Figure: A Plotly figure object for the prediction plot.
    """
    try:
        # Fetch historical stock data (defaults to 5y period and 1d interval)
        data = get_stock_data(ticker)

        if data is None or 'Close' not in data:
            return {"error": "Failed to fetch stock data or invalid data format."}, None

        # Reset index to make the date a column and rename columns for Prophet
        data = data.reset_index()  # Reset index to make date column explicit
        data.rename(columns={'index': 'Date'}, inplace=True)  # Ensure the date column is named 'Date'
        prophet_data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(prophet_data)

        # Create a DataFrame for future dates
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        # Create Plotly figure
        fig = go.Figure()

        # Plot historical closing prices
        fig.add_trace(go.Scatter(
            x=prophet_data['ds'],
            y=prophet_data['y'],
            mode='lines',
            name='Historical Closing Prices',
            line=dict(color='blue')
        ))

        # Plot predicted prices
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Predicted Prices',
            line=dict(color='green')
        ))

        # Plot confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Confidence Interval',
            line=dict(dash='dot', color='lightgreen')
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Confidence Interval',
            line=dict(dash='dot', color='lightgreen')
        ))

        # Customize layout
        fig.update_layout(
            title=f"{ticker.upper()} Stock Price Prediction (Prophet)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True
        )

        # Get the latest prediction
        latest_forecast = forecast[['ds', 'yhat']].iloc[-1]
        latest_predicted_price = latest_forecast['yhat']
        latest_date = latest_forecast['ds']

        # Return the figure and summary
        return {
            "latest_predicted_price": latest_predicted_price,
            "latest_date": latest_date
        }, fig

    except Exception as e:
        return {"error": str(e)}, None





def predict_and_plot_df(ticker, forecast_period=30):
    """
    Use the Prophet model to predict stock prices and return the forecast DataFrame.

    Args:
        ticker (str): Stock ticker symbol.
        forecast_period (int): Number of days to forecast (default is 30).

    Returns:
        pd.DataFrame: A DataFrame containing the forecasted stock prices.
    """
    try:
        # Fetch historical stock data (defaults to 5y period and 1d interval)
        data = get_stock_data(ticker)

        if data is None or 'Close' not in data:
            return {"error": "Failed to fetch stock data or invalid data format."}

        # Reset index to make the date a column and rename columns for Prophet
        data = data.reset_index()  # Reset index to make date column explicit
        data.rename(columns={'index': 'Date'}, inplace=True)  # Ensure the date column is named 'Date'
        prophet_data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(prophet_data)

        # Create a DataFrame for future dates
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        # Combine historical and forecasted data
        forecast_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Close'})
        prophet_data_new = prophet_data[['ds', 'y']].rename(columns={'ds': 'Date', 'y': 'Close'})
        #forecast_data_future = forecast_data[forecast_data['Date'] > pd.Timestamp.today()]
        combined_data = pd.concat([prophet_data_new, forecast_data]).reset_index(drop=True)
        # print(combined_data)
        return combined_data
        
        
        # Return the forecast DataFrame

    except Exception as e:
        return {"error": str(e)}