from flask import Flask, request, jsonify, render_template
from newsapi.newsapi_client import NewsApiClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import pandas as pd
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import io
import base64
from langdetect import detect

# Load environment variables
load_dotenv('.env')

# Initialize Flask app
app = Flask(__name__)

# Set up API key and sentiment analyzer
API_KEY = os.getenv('API_KEY')
analyzer = SentimentIntensityAnalyzer()

# Define sentiment analysis function

def sentiment_news_analysis(ticker_symbol):
    # Fetch news articles for the given ticker symbol
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

    # Generate visualizations
    plt.switch_backend('Agg')  # Required to avoid tkinter errors when Flask is running
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Sentiment Distribution Bar Chart
    sns.barplot(x=df['sentiment'].value_counts().index, y=df['sentiment'].value_counts().values, ax=axes[0])
    axes[0].set_title(f'Sentiment Distribution of {ticker_symbol} News')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Number of Titles')

    # Polarity Histogram
    sns.histplot(df['polarity'], kde=True, bins=30, color='blue', ax=axes[1])
    axes[1].set_title(f'Distribution of Polarity Scores for {ticker_symbol} News')
    axes[1].set_xlabel('Polarity Score')
    axes[1].set_ylabel('Frequency')

    # Compound Score Histogram
    sns.histplot(df['compound'], kde=True, bins=30, color='green', ax=axes[2])
    axes[2].set_title(f'Distribution of Compound Scores for {ticker_symbol} News')
    axes[2].set_xlabel('Compound Score')
    axes[2].set_ylabel('Frequency')

    # Pie Chart for Sentiment Proportion
    sentiment_counts = df['sentiment'].value_counts()
    axes[3].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[3].set_title(f'Sentiment Proportion of {ticker_symbol} News')

    # Save figure to a bytes buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return {"data": df.to_dict(orient='records'), "image": img_str}

# Flask route for analysis
@app.route('/sentiment', methods=['GET'])
def sentiment():
    ticker_symbol = request.args.get('ticker', default='MSFT', type=str)
    result = sentiment_news_analysis(ticker_symbol)
    if "error" in result:
        return jsonify(result)
    return render_template('sentiment.html', data=result['data'], image=result['image'])

# HTML Template Rendering
@app.route('/')
def home():
    return '''
    <html>
        <body>
            <h1>Sentiment News Analysis</h1>
            <form action="/sentiment" method="get">
                <label for="ticker">Enter Ticker Symbol:</label>
                <input type="text" id="ticker" name="ticker">
                <input type="submit" value="Analyze">
            </form>
        </body>
    </html>
    '''

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
