import os
import pandas as pd
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from langdetect import detect
from dotenv import load_dotenv
import requests
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv('.env')

# Set up API key and sentiment analyzer
API_KEY = os.getenv('API_KEY')
analyzer = SentimentIntensityAnalyzer()


def sentiment_news_analysis(ticker_symbol):
    """
    Perform sentiment analysis on news articles for the given ticker symbol.
    Returns a dictionary containing individual HTML strings for each Plotly graph.
    """
    try:
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

        # Generate Plotly visualizations
        sentiment_counts = df['sentiment'].value_counts()

        # Sentiment Distribution (Bar Chart)
        distribution_fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'positive': 'green', 'neutral': 'yellow', 'negative': 'red'},
            labels={'x': 'Sentiment', 'y': 'Count'},
            title=f'Sentiment Distribution<br>of {ticker_symbol}',
            width=500,
            height=400
        )
        distribution_html = distribution_fig.to_html(full_html=False)

        # Sentiment Proportion (Pie Chart)
        proportion_fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'positive': 'green', 'neutral': 'yellow', 'negative': 'red'},
            title=f'Sentiment Proportion<br>of {ticker_symbol}',
            width=500,
            height=400
        )
        proportion_html = proportion_fig.to_html(full_html=False)

        # Sentiment Summary (Pie Chart for Overall Sentiment)
        overall_sentiment = sentiment_counts.idxmax()
        summary_fig = go.Figure(
            go.Pie(
                labels=[f"Overall Sentiment: {overall_sentiment.capitalize()}"],
                values=[1],
                marker_colors=[{'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}[overall_sentiment]],
                textinfo="label",
                showlegend=False  # Hide legend
            )
        )
        summary_fig.update_layout(
            title=f"Overall Sentiment<br>of {ticker_symbol}",
            width=500,
            height=400
        )
        summary_html = summary_fig.to_html(full_html=False)

        return {
            "distribution_html": distribution_html,
            "proportion_html": proportion_html,
            "summary_html": summary_html
        }

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {"error": str(e)}
