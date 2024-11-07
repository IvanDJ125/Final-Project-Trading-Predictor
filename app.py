from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import plotly.graph_objs as go
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Fetch Polygon API key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')  # Render the input form

@app.route('/get-stock-info', methods=['POST'])
def get_stock_info():
    data = request.json
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    url = f"https://api.polygon.io/v1/meta/symbols/{ticker.upper()}/company"
    headers = {'Authorization': f'Bearer {POLYGON_API_KEY}'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        stock_info = response.json()
        relevant_info = {
            'name': stock_info.get('name', 'N/A'),
            'description': stock_info.get('description', 'No description available.'),
            'industry': stock_info.get('industry', 'N/A'),
            'sector': stock_info.get('sector', 'N/A'),
            'website': stock_info.get('website', 'N/A'),
            'ceo': stock_info.get('ceo', 'N/A'),
            'employees': stock_info.get('employees', 'N/A')
        }
        return jsonify(relevant_info)
    else:
        error_message = response.json().get('error', 'Unknown error occurred.')
        return jsonify({"error": f"Could not fetch stock info: {error_message}"}), response.status_code

@app.route('/get-historical-data', methods=['POST'])
def get_historical_data():
    data = request.json
    ticker = data.get('ticker')
    interval = data.get('interval', 'day')
    time_range = data.get('timeRange', '1d')

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    today = datetime.now()

    # Determine the start date based on the selected time range
    if time_range == '1d':
        from_date = today - timedelta(days=1)
    elif time_range == '1w':
        from_date = today - timedelta(weeks=1)
    elif time_range == '1m':
        from_date = today - timedelta(days=30)
    elif time_range == '3m':
        from_date = today - timedelta(days=90)
    elif time_range == '6m':
        from_date = today - timedelta(days=180)
    elif time_range == 'ytd':
        from_date = datetime(today.year, 1, 1)
    elif time_range == '1y':
        from_date = today - timedelta(days=365)
    elif time_range == '5y':
        from_date = today - timedelta(days=5 * 365)
    elif time_range == 'max':
        from_date = today - timedelta(days=365 * 10)  # Assuming a max period of 10 years; adjust as needed

    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = today.strftime('%Y-%m-%d')

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/{interval}/{from_date_str}/{to_date_str}"
    headers = {'Authorization': f'Bearer {POLYGON_API_KEY}'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and data['results']:
            results = data['results']
            processed_data = []
            for result in results:
                processed_data.append({
                    'time': result['t'],
                    'open': result['o'],
                    'close': result['c'],
                    'high': result['h'],
                    'low': result['l']
                })
            return jsonify(processed_data)
        else:
            return jsonify({"error": "No historical data available for the given ticker and interval."}), 404
    else:
        error_message = response.json().get('error', 'Unknown error occurred.')
        return jsonify({"error": f"Could not fetch historical data: {error_message}"}), response.status_code

@app.route('/plot-historical-data', methods=['POST'])
def plot_historical_data():
    data = request.json
    if not data:
        return jsonify({"error": "No historical data to plot."}), 400

    # Extracting closing prices and timestamps
    times = [result['time'] for result in data]
    closing_prices = np.array([result['close'] for result in data])
    opening_prices = [result['open'] for result in data]
    high_prices = [result['high'] for result in data]
    low_prices = [result['low'] for result in data]

    # Convert timestamps to readable dates
    dates = [datetime.fromtimestamp(ts / 1000) for ts in times]

    # Calculate Bollinger Bands
    window = 10  # Standard period for Bollinger Bands
    if len(closing_prices) < window:
        return jsonify({"error": "Not enough data to calculate Bollinger Bands."}), 400

    middle_band = np.convolve(closing_prices, np.ones(window) / window, mode='valid')
    std_dev = np.array([np.std(closing_prices[i:i + window]) for i in range(len(closing_prices) - window + 1)])
    upper_band = middle_band + (std_dev * 2)
    lower_band = middle_band - (std_dev * 2)

    # Padding to align with candlestick data
    padding = [np.nan] * (window - 1)
    upper_band = np.concatenate((padding, upper_band))
    lower_band = np.concatenate((padding, lower_band))
    middle_band = np.concatenate((padding, middle_band))

    # Create a Plotly figure with candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dates,
                                  open=opening_prices,
                                  close=closing_prices,
                                  high=high_prices,
                                  low=low_prices,
                                  name='Candlestick'))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=dates, y=upper_band, line=dict(color='gray', width=1), name='Upper Band'))
    fig.add_trace(go.Scatter(x=dates, y=lower_band, line=dict(color='gray', width=1), name='Lower Band'))
    fig.add_trace(go.Scatter(x=dates, y=middle_band, line=dict(color='gray', width=1), name='Middle Band (SMA)'))

    # Add titles and labels
    fig.update_layout(title='Historical Prices with Bollinger Bands',
                      xaxis_title='Date',
                      yaxis_title='Price ($)',
                      template='plotly_white')

    # Convert the figure to JSON
    graphJSON = fig.to_json()  # Use Plotly's to_json method for direct conversion

    return jsonify(graphJSON)

@app.route('/plot-scatter-data', methods=['POST'])
def plot_scatter_data():
    data = request.json
    if not data:
        return jsonify({"error": "No historical data to plot."}), 400

    # Extract closing prices and timestamps for scatter plot
    times = [result['time'] for result in data]
    closing_prices = [result['close'] for result in data]

    # Convert timestamps to readable dates
    dates = [datetime.fromtimestamp(ts / 1000) for ts in times]

    # Create a Plotly figure for scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closing_prices, mode='markers+lines', name='Closing Price'))

    # Add titles and labels
    fig.update_layout(title='Historical Closing Prices (Scatter Plot)',
                      xaxis_title='Date',
                      yaxis_title='Closing Price ($)',
                      template='plotly_white')

    # Convert the figure to JSON
    graphJSON = fig.to_json()  # Use Plotly's to_json method for direct conversion

    return jsonify(graphJSON)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
