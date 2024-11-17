from flask import Flask, request, jsonify, render_template
from newsapi.newsapi_client import NewsApiClient
from dotenv import load_dotenv
import os
import pandas as pd
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from langdetect import detect
from datetime import datetime, timedelta

# Load environment variables
load_dotenv('.env')

# Initialize Flask app
app = Flask(__name__)

# Set up API key and sentiment analyzer
API_KEY = os.getenv('API_KEY')
analyzer = SentimentIntensityAnalyzer()

# Define sentiment analysis function
def sentiment_data(ticker_symbol):
    # Fetch weekly articles for the ticker symbol
    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    from_date = one_week_ago.strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')

    url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&from={from_date}&to={to_date}&sortBy=popularity&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Failed to retrieve data", "status_code": response.status_code}

    # Parse the JSON data and normalize it
    data = response.json()
    articles = data.get('articles', [])
    df = pd.json_normalize(articles)

    # Filter for required columns and rename them
    df = df[['source.name', 'title', 'publishedAt']]
    df.rename(columns={'source.name': 'source', 'publishedAt': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%m/%d')

    # Filter only English titles
    def is_english(text):
        try:
            return detect(text) == 'en'
        except:
            return False

    df = df[df['title'].apply(is_english)]

    # Preprocess text
    def preprocess_text(text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#|\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    df['title'] = df['title'].apply(preprocess_text)

    # Analyze sentiment
    def analyze_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        vader_score = analyzer.polarity_scores(text)
        compound = vader_score['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return sentiment, polarity, compound

    df[['sentiment', 'polarity', 'compound']] = df['title'].apply(lambda x: analyze_sentiment(x)).apply(pd.Series)
    return df

# Flask route for analysis
@app.route('/sentiment', methods=['GET'])
def sentiment():
    ticker_symbol = request.args.get('ticker', default='MSFT', type=str)
    result = sentiment_data(ticker_symbol)
    if isinstance(result, dict) and "error" in result:
        return jsonify(result)
    return render_template('sentiment_data.html', data=result.to_dict('records'))

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('sentiment_data.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
