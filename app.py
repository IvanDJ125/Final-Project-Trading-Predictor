from flask import Flask, render_template, request
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__)

def get_stock_info(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # Format market cap
        raw_market_cap = info.get("marketCap", 0)  # Default to 0 if not available
        formatted_market_cap = format_market_cap(raw_market_cap)

        stock_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Address": info.get("address1", "N/A"),
            "City": info.get("city", "N/A"),
            "State": info.get("state", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": formatted_market_cap,
            "Stock Exchange": info.get("exchange", "N/A"),
            "Business Summary": info.get("longBusinessSummary", "N/A"),
            "Full-Time Employees": info.get("fullTimeEmployees", "N/A")
        }

        return stock_info
    except Exception as e:
        return {"Error": str(e)}

def format_market_cap(market_cap):
    if market_cap >= 1_000_000_000_000:  # Trillions
        return f"${market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:  # Billions
        return f"${market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:  # Millions
        return f"${market_cap / 1_000_000:.2f}M"
    else:  # Smaller numbers
        return f"${market_cap:.2f}"

def get_stock_data(ticker_symbol, period, interval):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period, interval=interval)

        if hist.empty:
            return None, "No data available for the selected period and interval."

        # Reset the index to include the date
        hist.reset_index(inplace=True)

        return hist, None
    except Exception as e:
        return None, str(e)

def generate_stock_chart(hist_data, ticker_symbol, period, interval, graph_type):
    fig = go.Figure()

    if graph_type in ["candlestick", "both"]:
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=hist_data["Datetime"] if "Datetime" in hist_data.columns else hist_data["Date"],
                open=hist_data["Open"],
                high=hist_data["High"],
                low=hist_data["Low"],
                close=hist_data["Close"],
                name="Candlestick"
            )
        )

    if graph_type in ["closing_price", "both"]:
        # Add closing price line
        fig.add_trace(
            go.Scatter(
                x=hist_data["Datetime"] if "Datetime" in hist_data.columns else hist_data["Date"],
                y=hist_data["Close"],
                mode="lines",
                name="Closing Price",
                line=dict(color="blue", width=1.5),
            )
        )

    fig.update_layout(
        title=f"{ticker_symbol} Stock Price ({period} | {interval})",
        xaxis_title="Datetime" if "Datetime" in hist_data.columns else "Date",
        yaxis_title="Price",
        template="plotly_white",
    )

    # Render the chart to HTML
    chart_html = fig.to_html(full_html=False)
    return chart_html

def perform_sma_analysis(ticker_symbol, period, interval):
    # Ensure minimum period and interval
    valid_periods = ["6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    valid_intervals = ["1d", "5d", "1wk", "1mo", "3mo"]

    if period not in valid_periods:
        return None, "Invalid period selected for SMA Analysis. Minimum period is 6 months.", None, None

    if interval not in valid_intervals:
        return None, "Invalid interval selected for SMA Analysis. Minimum interval is 1 day.", None, None

    # Fetch historical data
    hist_data, error = get_stock_data(ticker_symbol, period, interval)
    if error:
        return None, error, None, None

    # Calculate SMAs
    hist_data['SMA_5'] = hist_data['Close'].rolling(window=5).mean()
    hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()

    # Extract the last row
    last_row = hist_data[['Close', 'SMA_5', 'SMA_20', 'SMA_50']].tail(1).iloc[0]
    last_close = last_row['Close']
    last_sma5 = last_row['SMA_5']
    last_sma20 = last_row['SMA_20']
    last_sma50 = last_row['SMA_50']

    # Conditional formatting and opinion
    if last_close > last_sma5 and last_sma5 > last_sma20 and last_sma20 > last_sma50:
        opinion = "Strong Bullish Signal: The stock is in a strong uptrend."
    elif last_close < last_sma5 and last_sma5 < last_sma20 and last_sma20 < last_sma50:
        opinion = "Strong Bearish Signal: The stock is in a strong downtrend."
    elif last_close > last_sma20 and last_close > last_sma50:
        opinion = "Moderate Bullish Signal: The stock is above medium-term trends."
    elif last_close < last_sma20 and last_close < last_sma50:
        opinion = "Moderate Bearish Signal: The stock is below medium-term trends."
    else:
        opinion = "Neutral Signal: The stock is trading within mixed trend signals."

    # Generate SMA plot
    fig = go.Figure()

    # Plot the closing price
    fig.add_trace(go.Scatter(
        x=hist_data["Datetime"] if "Datetime" in hist_data.columns else hist_data["Date"],
        y=hist_data['Close'],
        mode='lines',
        name='Closing Price',
        line=dict(color='blue')
    ))

    # Plot SMA 5
    fig.add_trace(go.Scatter(
        x=hist_data["Datetime"] if "Datetime" in hist_data.columns else hist_data["Date"],
        y=hist_data['SMA_5'],
        mode='lines',
        name='SMA 5',
        line=dict(dash='dot', color='orange')
    ))

    # Plot SMA 20
    fig.add_trace(go.Scatter(
        x=hist_data["Datetime"] if "Datetime" in hist_data.columns else hist_data["Date"],
        y=hist_data['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(dash='dot', color='green')
    ))

    # Plot SMA 50
    fig.add_trace(go.Scatter(
        x=hist_data["Datetime"] if "Datetime" in hist_data.columns else hist_data["Date"],
        y=hist_data['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(dash='dot', color='red')
    ))

    # Customize the layout
    fig.update_layout(
        title=f"{ticker_symbol} Closing Price and SMAs (5, 20, 50)",
        xaxis_title="Datetime" if "Datetime" in hist_data.columns else "Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True,
        template="plotly_white",
    )

    # Render the chart to HTML
    sma_chart_html = fig.to_html(full_html=False)

    # Prepare the last row data
    last_row_data = last_row.to_dict()

    return sma_chart_html, None, opinion, last_row_data

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_data = None
    error = None
    chart_html = None
    sma_chart_html = None
    sma_opinion = None
    sma_last_row = None
    ticker_symbol = None
    period = "1mo"  # Default period
    interval = "1d"  # Default interval
    sma_period = "6mo"  # Default SMA period
    sma_interval = "1d"  # Default SMA interval
    graph_type = "both"  # Default graph type

    if request.method == 'POST':
        action = request.form.get('action')
        ticker_symbol = request.form.get('ticker', '').upper()
        period = request.form.get('period', period)
        interval = request.form.get('interval', interval)
        graph_type = request.form.get('graph_type', graph_type)
        sma_period = request.form.get('sma_period', sma_period)
        sma_interval = request.form.get('sma_interval', sma_interval)

        if action == 'search':
            if ticker_symbol:
                stock_data = get_stock_info(ticker_symbol)
                if "Error" in stock_data:
                    error = stock_data["Error"]
                    stock_data = None

        elif action == 'graph':
            if ticker_symbol and period and interval:
                hist_data, hist_error = get_stock_data(ticker_symbol, period, interval)
                if hist_error:
                    error = hist_error
                elif hist_data is not None:
                    chart_html = generate_stock_chart(hist_data, ticker_symbol, period, interval, graph_type)

            # Ensure stock data persists after graphing
            if not stock_data:
                stock_data = get_stock_info(ticker_symbol)

        elif action == 'sma_analysis':
            if ticker_symbol and sma_period and sma_interval:
                sma_chart_html, sma_error, sma_opinion, sma_last_row = perform_sma_analysis(ticker_symbol, sma_period, sma_interval)
                if sma_error:
                    error = sma_error

            # Ensure stock data persists after SMA analysis
            if not stock_data:
                stock_data = get_stock_info(ticker_symbol)

    return render_template(
        'index.html',
        stock_data=stock_data,
        error=error,
        chart_html=chart_html,
        sma_chart_html=sma_chart_html,
        sma_opinion=sma_opinion,
        sma_last_row=sma_last_row,
        ticker_symbol=ticker_symbol,
        period=period,
        interval=interval,
        sma_period=sma_period,
        sma_interval=sma_interval,
        graph_type=graph_type
    )

if __name__ == '__main__':
    app.run(debug=True)
