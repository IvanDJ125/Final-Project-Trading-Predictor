import os
import sys

# Add the project root directory to sys.path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import pandas as pd
import plotly.graph_objects as go
from utils.stock_data import get_stock_data  # Import fetch_stock_data from the same module

def calculate_smas_and_opinion(ticker, plot=False):
    """
    Calculate SMAs (20, 50) and provide an opinion based on the SMA strategy.

    Args:
        ticker (str): Stock ticker symbol.
        plot (bool): Whether to plot the SMAs and closing price using Plotly.

    Returns:
        dict: A dictionary containing the last row of data, calculated SMAs, and an opinion.
        plotly.graph_objects.Figure: Optional Plotly figure if `plot` is True.
    """
    try:
        # Fetch historical stock data
        df = get_stock_data(ticker)

        if df is None or 'Close' not in df:
            return {"error": "Failed to fetch stock data or invalid data format."}

        # Calculate SMAs
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Extract the last row
        last_row = df[['Close', 'SMA_20', 'SMA_50']].tail(1)
        last_close = last_row['Close'].values[0]
        last_sma20 = last_row['SMA_20'].values[0]
        last_sma50 = last_row['SMA_50'].values[0]

        # Conditional formatting and opinion
        if last_close > last_sma20 and last_sma20 > last_sma50:
            opinion = "Strong Bullish Signal: The stock is in an uptrend, and the closing price is above the SMA 20."
        elif last_sma20 > last_sma50 and last_close < last_sma20:
            opinion = "Moderate Bullish Signal: The stock is in an uptrend, but the closing price is below the SMA 20, indicating short-term weakness."
        elif last_close < last_sma50 and last_sma20 < last_sma50:
            opinion = "Strong Bearish Signal: The stock is in a downtrend, and the closing price is below the SMA 50."
        elif last_sma20 < last_sma50 and last_close > last_sma20:
            opinion = "Moderate Bearish Signal: The stock is in a downtrend, but the closing price is above the SMA 20, indicating potential short-term recovery."
        else:
            opinion = "Neutral Signal: The stock is trading sideways or lacks a clear trend."

        # Create the Plotly graph if requested
        fig = None
        if plot:
            fig = go.Figure()

            # Plot the closing price
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Closing Price',
                line=dict(color='blue')
            ))

            # Plot SMA 20
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(dash='dot', color='green')
            ))

            # Plot SMA 50
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(dash='dot', color='red')
            ))

            # Customize the layout
            fig.update_layout(
                title=f"{ticker.upper()} Closing Price and SMAs (20, 50)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=True
            )

        # Return the last row data and opinion
        return {
            "last_row": last_row.to_dict(orient='records')[0],
            "opinion": opinion,
            "plot": fig
        }

    except Exception as e:
        return {"error": str(e)}



def calculate_and_plot_rsi(ticker, window_length=14, plot=True):
    """
    Calculate the Relative Strength Index (RSI), plot it, and provide an opinion.

    Args:
        ticker (str): Stock ticker symbol.
        window_length (int): Period for calculating RSI (default is 14).
        plot (bool): Whether to plot the RSI using Plotly.

    Returns:
        dict: A dictionary containing the RSI values, last RSI value, and opinion.
        plotly.graph_objects.Figure: Optional Plotly figure if `plot` is True.
    """
    try:
        # Fetch historical stock data
        df = get_stock_data(ticker)

        if df is None or 'Close' not in df:
            return {"error": "Failed to fetch stock data or invalid data format."}

        # Calculate RSI
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Extract the last row with the RSI value
        last_row = df[['Close', 'RSI']].tail(1)
        last_close = last_row['Close'].values[0]
        last_rsi = last_row['RSI'].values[0]

        # Generate opinion based on RSI
        if last_rsi > 70:
            opinion = "The stock is overbought. RSI is above 70."
        elif last_rsi < 30:
            opinion = "The stock is oversold. RSI is below 30."
        else:
            opinion = "The stock is in a neutral zone. RSI is between 30 and 70."

        # Create Plotly figure if requested
        fig = None
        if plot:
            fig = go.Figure()

            # Plot the RSI line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='blue')
            ))

            # Add overbought level (70)
            fig.add_shape(
                type='line',
                x0=df.index[0],
                y0=70,
                x1=df.index[-1],
                y1=70,
                line=dict(color='red', dash='dash'),
                name='Overbought (70)'
            )

            # Add oversold level (30)
            fig.add_shape(
                type='line',
                x0=df.index[0],
                y0=30,
                x1=df.index[-1],
                y1=30,
                line=dict(color='green', dash='dash'),
                name='Oversold (30)'
            )

            # Customize layout
            fig.update_layout(
                title=f"{ticker.upper()} Relative Strength Index (RSI)",
                xaxis_title="Date",
                yaxis_title="RSI",
                xaxis_rangeslider_visible=True,
                yaxis=dict(range=[0, 100])  # RSI ranges from 0 to 100
            )

        # Return results and the plot
        return {
            "last_row": last_row.to_dict(orient='records')[0],
            "opinion": opinion,
            "plot": fig
        }

    except Exception as e:
        return {"error": str(e)}


def calculate_and_plot_macd(ticker, short_window=12, long_window=26, signal_window=9, plot=True):
    """
    Calculate the MACD, Signal Line, and Histogram, and provide an opinion.

    Args:
        ticker (str): Stock ticker symbol.
        short_window (int): EMA short window (default is 12).
        long_window (int): EMA long window (default is 26).
        signal_window (int): Signal line window (default is 9).
        plot (bool): Whether to plot the MACD and Signal Line using Plotly.

    Returns:
        dict: A dictionary containing the MACD values, Signal Line, and opinion.
        plotly.graph_objects.Figure: Optional Plotly figure if `plot` is True.
    """
    try:
        # Fetch historical stock data
        df = get_stock_data(ticker)

        if df is None or 'Close' not in df:
            return {"error": "Failed to fetch stock data or invalid data format."}

        # Calculate MACD and related values
        df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal_Line']

        # Get the last row
        last_row = df[['Close', 'MACD', 'Signal_Line']].tail(1)
        last_close = last_row['Close'].values[0]
        last_macd = last_row['MACD'].values[0]
        last_signal = last_row['Signal_Line'].values[0]

        # Provide advice based on MACD and Signal Line
        if last_macd > 0 and last_macd > last_signal:
            opinion = "Bullish signal: Positive MACD above Signal Line, indicating upward momentum."
        elif last_macd > 0 and last_macd < last_signal:
            opinion = "Bullish trend weakening: Positive MACD below Signal Line, indicating slowing momentum."
        elif last_macd < 0 and last_macd < last_signal:
            opinion = "Bearish signal: Negative MACD below Signal Line, indicating downward momentum."
        elif last_macd < 0 and last_macd > last_signal:
            opinion = "Bearish trend weakening: Negative MACD above Signal Line, indicating slowing downward momentum."
        else:
            opinion = "Potential trend reversal: MACD is near the Signal Line, indicating a crossover."

        # Create Plotly figure if requested
        fig = None
        if plot:
            fig = go.Figure()

            # Plot MACD
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ))

            # Plot Signal Line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Signal_Line'],
                mode='lines',
                name='Signal Line',
                line=dict(dash='dot', color='orange')
            ))

            # Plot Histogram
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Histogram'],
                name='Histogram',
                marker_color='gray'
            ))

            # Customize the layout
            fig.update_layout(
                title=f"{ticker.upper()} MACD, Signal Line, and Histogram",
                xaxis_title="Date",
                yaxis_title="Value",
                xaxis_rangeslider_visible=True
            )

        # Return results and the plot
        return {
            "last_row": last_row.to_dict(orient='records')[0],
            "opinion": opinion,
            "plot": fig
        }

    except Exception as e:
        return {"error": str(e)}
