from flask import Flask, request, jsonify, render_template
from newsapi.newsapi_client import NewsApiClient
import dotenv
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import pandas as pd
import re
import textblob
from textblob import TextBlob
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import io
import base64
import langdetect
from langdetect import detect
import numpy as np


# Load environment variables
load_dotenv('.env')

# Initialize Flask app
app = Flask(__name__)

# Set up API key and sentiment analyzer
API_KEY = os.getenv('API_KEY')
analyzer = SentimentIntensityAnalyzer()

# Define sentiment analysis function
def sentiment_news_analysis(ticker_symbol):
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

 # Generate visualizations
# Define color legend
    legend_labels = [
    'Positive (>= 0.05)',
    'Neutral (-0.05 to 0.05)',
    'Negative (<= -0.05)'
    ]
    legend_colors = ['green', 'gold', 'red']
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]

# Set up the 1x3 layout for all the visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Sentiment Distribution Bar Chart
    sns.barplot(x=df['sentiment'].value_counts().index, 
            y=df['sentiment'].value_counts().values, 
            ax=axes[0], 
            palette=legend_colors)
    axes[0].set_title(f'Sentiment Distribution of {ticker_symbol} News')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Number of Titles')
    axes[0].tick_params(axis='x', rotation=45)

# Pie Chart for Sentiment Proportion
    sentiment_counts = df['sentiment'].value_counts()
    axes[1].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=legend_colors)
    axes[1].set_title(f'Sentiment Proportion of {ticker_symbol} News')

# Average Compound Score Gauge-like Diagram
# Step 1: Calculate the average compound score
    avg_compound = df['compound'].mean()

# Step 2: Determine overall sentiment and set color
    if avg_compound >= 0.05:
     overall_sentiment = 'Positive'
     color = 'green'
    elif avg_compound <= -0.05:
        overall_sentiment = 'Negative'
        color = 'red'
    else:
        overall_sentiment = 'Neutral'
        color = 'gold'

# Step 3: Create a gauge-like graphic
    wedges, _ = axes[2].pie([1], colors=[color], radius=1)

# Add an annotation to indicate the overall sentiment
    axes[2].text(0, 0.1, overall_sentiment, ha='center', va='center', fontsize=24, fontweight='bold', color=color)

# Add the phrase to state the overall sentiment clearly
    axes[2].text(0, 0, f"Overall {ticker_symbol} news are {overall_sentiment}.", 
             ha='center', va='center', fontsize=12, color='black')

    axes[2].set_title(f'Overall Sentiment for {ticker_symbol} News')

# Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Adjust layout and save figure to buffer
    plt.tight_layout()
    buffer = io.BytesIO()
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

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('sentiment.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
