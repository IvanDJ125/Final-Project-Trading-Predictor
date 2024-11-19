import plotly.graph_objects as go
from utils.stock_data import get_stock_data

def plot_closing_prices(ticker):
    """
    Fetch and plot the closing price data for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object for the closing prices.
        dict: A summary of the data, including the latest closing price.
    """
    try:
        # Fetch stock data
        df = get_stock_data(ticker)

        if df is None or 'Close' not in df:
            return {"error": "Failed to fetch stock data or invalid data format."}, None

        # Create the Plotly figure
        fig = go.Figure()

        # Plot the closing price
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Closing Price',
            line=dict(color='blue')
        ))

        # Customize the layout
        fig.update_layout(
            title=f"{ticker.upper()} Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True
        )

        # Get the latest closing price
        latest_close = df['Close'].iloc[-1]
        latest_date = df.index[-1]

        # Return the figure and summary
        return {
            "latest_close": latest_close,
            "latest_date": latest_date
        }, fig

    except Exception as e:
        return {"error": str(e)}, None
