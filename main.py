import os
import sys
from flask import Flask, render_template, request, jsonify

# Add the project folder to sys.path for imports
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

from flask import Flask, render_template, request, jsonify
from utils.stock_data import get_stock_info
from utils.closing_price import plot_closing_prices
from utils.indicators import calculate_smas_and_opinion, calculate_and_plot_rsi, calculate_and_plot_macd
from utils.prophet_model import predict_and_plot_prophet


app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the homepage with the search form.
    """
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """
    Process the stock ticker input and return all calculations and plots.
    """
    try:
        ticker = request.form['ticker'].upper()

        # Fetch stock information
        stock_info = get_stock_info(ticker)
        if stock_info is None:
            return jsonify({"error": f"Failed to fetch stock information for ticker {ticker}."}), 400

        # Fetch and plot closing prices
        closing_prices_result, closing_prices_fig = plot_closing_prices(ticker)

        if "error" in closing_prices_result:
            return jsonify({"error": closing_prices_result["error"]}), 400

        # Calculate SMA and opinion
        sma_result = calculate_smas_and_opinion(ticker, plot=True)
        if "error" in sma_result:
            return jsonify({"error": sma_result["error"]}), 400

        # Calculate MACD and opinion
        rsi_result = calculate_and_plot_rsi(ticker, plot=True)
        if "error" in rsi_result:
            return jsonify({"error": rsi_result["error"]}), 400

        # Calculate Bollinger Bands and opinion
        macd_result = calculate_and_plot_macd(ticker, plot=True)
        if "error" in macd_result:
            return jsonify({"error": macd_result["error"]}), 400

        # Use Prophet for predictions
        prophet_result, prophet_fig = predict_and_plot_prophet(ticker, forecast_period=60)
        if "error" in prophet_result:
            return jsonify({"error": prophet_result["error"]}), 400

        # Combine all results and plots into a single response
        return jsonify({
            "stock_info": stock_info,
            "closing_prices": closing_prices_result,
            "closing_prices_plot": closing_prices_fig.to_html(full_html=False),
            "sma_opinion": sma_result["opinion"],
            "sma_plot": sma_result["plot"].to_html(full_html=False),
            "rsi_opinion": rsi_result["opinion"],
            "rsi_plot": rsi_result["plot"].to_html(full_html=False),
            "macd_opinion": macd_result["opinion"],
            "macd_plot": macd_result["plot"].to_html(full_html=False),
            "prophet_prediction": {
                "latest_price": prophet_result["latest_predicted_price"],
                "date": prophet_result["latest_date"]
            },
            "prophet_plot": prophet_fig.to_html(full_html=False),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
