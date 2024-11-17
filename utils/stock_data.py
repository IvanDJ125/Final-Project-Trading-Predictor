import yfinance as yf

def format_market_cap(market_cap):
    """
    Format market capitalization
    """
    if market_cap >= 1_000_000_000_000:  # Trillions
        return f"${market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:  # Billions
        return f"${market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:  # Millions
        return f"${market_cap / 1_000_000:.2f}M"
    else:  # Smaller numbers
        return f"${market_cap:.2f}"

def get_stock_info(ticker):
    """
    Fetch stock information for a given ticker and return a formatted dictionary.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Format market capitalization
        market_cap = info.get("marketCap", 0)
        formatted_market_cap = format_market_cap(market_cap) if market_cap else "N/A"

        # Create a dictionary of stock information
        stock_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": formatted_market_cap,
            "Stock Exchange": info.get("exchange", "N/A"),
            "Full-Time Employees": info.get("fullTimeEmployees", "N/A"),
            "Business Summary": info.get("longBusinessSummary", "N/A")
        }
        return stock_info

    except Exception as e:
        print(f"Error fetching stock info for ticker {ticker}: {e}")
        return None

def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetch historical stock data for a given ticker, period, and interval.
    """
    try:
        # Fetch historical stock data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker} with period '{period}' and interval '{interval}'.")
        
        data.index = data.index.date

        return data

    except Exception as e:
        print(f"Error fetching stock data for ticker {ticker}: {e}")
        return None

