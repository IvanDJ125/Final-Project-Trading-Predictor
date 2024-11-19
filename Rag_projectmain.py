#!/usr/bin/env python
# coding: utf-8

# In[24]:



# Import Dependencies
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import openai
import sys
import numpy as np
from openai import OpenAI
import os
import base64
import io
import json
from tiktoken import get_encoding
from dotenv import load_dotenv
#%pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
import pdfkit
import pandas as pd
from transformers import GPT2Tokenizer
import yfinance as yf
import pandas as pd
import os
from fpdf import FPDF
import feedparser
import ssl
from datetime import datetime, timedelta
from utils.sentiment_analysis import sentiment_news_analysis, TextBlob, detect
import re



# ### Get Stock Financials and Overview

# In[25]:

ticker_symbol = input("Input ticker symbol: ").strip()



# Get company financials
ticker = yf.Ticker(ticker_symbol)
#stock_data = yf.download(ticker_symbol, start="2024-01-01", end="2024-11-01")

if ticker != '*USD':
    income_statement = ticker.financials
    balance_sheet = ticker.balance_sheet
    cash_flow = ticker.cashflow
    company_info = ticker.info
    price_data = ticker.history(period="2y")  # Options include '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    
    
else: #get crypyo info
    price_data = ticker.history(period="2y")




def save_crypto_data_to_pdf(ticker):
    if 'usd' in ticker.lower():
        price_data_str = price_data.to_string()
        return str(price_data_str)
    else:
        return None
   

analyzer = SentimentIntensityAnalyzer()

def sentiment_news_analysis(ticker_symbol):

    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    
    # Fetch news articles for the given ticker symbol
    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    from_date = one_week_ago.strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')

    url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&from={from_date}&to={to_date}&sortBy=popularity&apiKey={API_KEY}"
    response = requests.get(url)
    # print(response.reason)

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

    
    buffer = io.BytesIO()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return {"data": df.to_dict(orient='records'), "image": img_str}

# Testing NLP 
# print(sentiment_news_analysis(ticker_symbol))

# import sys sys.exit()


print(predict_and_plot_df(ticker_symbol, forecast_period=30))



# import sys
# sys.exit()




def save_financial_data_to_string(ticker):
    # Fetch financial data using yfinance
    company = yf.Ticker(ticker)
    # price_data = company.history(period="1y").astype(str).to_string()

    
    if 'usd' not in ticker.lower(): 
        income_statement = company.financials.astype(str).to_string()
        balance_sheet = company.balance_sheet.astype(str).to_string()
        cash_flow = company.cashflow.astype(str).to_string()
        company_overview = company.info
        sentiment_analysis_df = json.dumps(sentiment_news_analysis(ticker), indent=4)
        price_data_str = company.history(period="2y").astype(str).to_string()
        # prediction_str = predict_and_plot_df(ticker, forecast_period=30)
        # predictions_df = predict_and_plot_df(ticker_symbol, forecast_period=30)
        # data, columns = predictions_df
        # predictions_df = pd.DataFrame(predict_and_plot_df(ticker_symbol, forecast_period=30))
        # predictions_df = predictions_df.applymap(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else x)
        # predictions_df_str = json.dumps(predictions_df.to_dict(orient='records'), indent=4)
        # predictions_df_str = predictions_df.to_string()
        company_overview_str = ""
        # company_overview_str = f"Address: {company_overview[0]}, {company_overview['city']}, {company_overview['state']}, {company_overview['zip']}, {company_overview['country']}\n"
        company_overview_str += f"Website: {company_overview['website']}\n"
        company_overview_str += f"Industry: {company_overview['industry']}\n"
        company_overview_str += f"Sector: {company_overview['sector']}\n"
        company_overview_str += f"Business Summary: {company_overview['longBusinessSummary']}\n"


        complete_company_report = ""

        complete_company_report += company_overview_str
        complete_company_report += income_statement
        complete_company_report += balance_sheet
        complete_company_report += cash_flow
        complete_company_report += price_data_str
        complete_company_report += sentiment_analysis_df
        # complete_company_report += predictions_df_str
    else:
        price_data = company.history(period="2y").astype(str).to_string()
        return price_data
    

    return str(complete_company_report)
    
  

save_financial_data_to_string(ticker_symbol)

# sys.exit()
# In[11]:


def save_crypto_data_to_string(ticker):
    company = yf.Ticker(ticker)
    if 'usd' in ticker.lower():
        price_data_str = company.history(period="2y").astype(str).to_string()
        return str(price_data_str)
    else:
        return None
   

save_crypto_data_to_string(ticker_symbol)


# In[13]:




if 'usd' in ticker_symbol:
    save_crypto_data_to_string(ticker_symbol)
else:
    save_financial_data_to_string(ticker_symbol)



# ## Create RAG

# #### Load OPEN AI key

# In[14]:



# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API Key from environment variable
# Open AI (LLM to process text)
openai.api_key = os.getenv("OPENAI_KEY_API")
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY_API"))



# In[15]:


# Function to generate embeddings using the new OpenAI API
def get_embedding(text, engine="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=engine)
    embedding = response.data[0].embedding
    embedding = np.array(embedding).reshape(1, -1)  # Ensure the embedding is a 2D array
    return response.data[0].embedding


# In[16]:


def upload_chunks_to_local_memory(text_chunks, delay_seconds=2):
    for idx, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk, delay_seconds=delay_seconds)
        local_vector_store[f"chunk_{idx}"] = {
            "embedding": embedding,
            "text": chunk
        }
        time.sleep(delay_seconds) 


# ### Add Vector Database to Local Storage

# In[17]:


# Dictionary to store vectors in local memory
local_vector_store = {}

# Function to add chunks to the local vector store
def upload_chunks_to_local_memory(text_chunks):
    for idx, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        local_vector_store[f"chunk_{idx}"] = {
            "embedding": embedding,
            "text": chunk
        }




# Function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=1024):
    encoding = get_encoding("cl100k_base")  # Tokenizer model
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [encoding.decode(chunk) for chunk in chunks]
    return text_chunks


# In[20]:


def ask_question(question):
    # Get the embedding of the question
    question_embedding = get_embedding(question)
    
    # Verify that the embedding has the correct number of dimensions (1536)
    if len(question_embedding) != 1536:
        raise ValueError(f"Embedding size is incorrect: {len(question_embedding)} dimensions found, expected 1536.")
    
    # Reshape the question embedding to 2D
    question_embedding = np.array(question_embedding).reshape(1, -1)
    
    # Query the local vector store
    result = sorted(
        local_vector_store.values(),
        key=lambda x: cosine_similarity(np.array(x['embedding']).reshape(1, -1), question_embedding)[0][0],
        reverse=True
    )[:5]
    
    # Use OpenAI to generate an answer based on retrieved chunks
    context = " ".join([match['text'] for match in result])
    
    # Updated OpenAI API call for chat models using `ChatCompletion.create` method
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": f"You are a Hedge Fund manager offering investment price and outlook analysis and recommendations. Based on the following context: {context}, provide a complete and concise answer to the question: {question}. Ensure your response forms a full, coherent thought and does not trail off, staying within the character limit."}

    #     ],
    #     max_tokens=200
    # )
    # Get today's date
    today_date = datetime.now().strftime("%Y-%m-%d")

   

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                f"Always mention the current price of the stock first as of the most recent closing date."
                f"Always finish the last sentence of the response."
                f"You are a Hedge Fund manager offering investment price and outlook analysis and outlook without using the word recommendation. "
                f"The current date is {today_date}. Based on the following context: {context}, "
                f"provide a complete and concise answer to the question: {question} without trailing off at the end. "
                f"The pricing data is in ascending order based on the date and the date is in YYYY-MM-DD format."
                f"Always include news and sentiment analysis and sentiment score as part of the analysis. Include the news impacting the sentiment score."
                "Ensure your response forms a full, coherent thought and does not trail off, staying within the character limit."
            )
        }
    ],
    max_tokens=250,
    temperature=0.2
)
        
    return response.choices[0].message.content

    
    


# In[21]:


# Extract text from the PDF

# folder_path = os.path.join(os.getcwd(), ticker_symbol)
if 'usd' in ticker_symbol:
    pdf_file_path = save_crypto_data_to_string(ticker_symbol)
    # pdf_text = save_crypto_data_to_pstringticker_symbol)
else:
    pdf_file_path = save_financial_data_to_string(ticker_symbol)
    # pdf_text = extract_text_from_pdf(pdf_file_path)

# Chunk the text
chunks = chunk_text(pdf_file_path)

# Upload the chunks to Pinecone
# upload_chunks_to_pinecone(chunks)

# Upload Chunks to local store
upload_chunks_to_local_memory(chunks)


# In[22]:



question = input(f"Ask me about {ticker_symbol}: ")
answer = ask_question(question)


# In[ ]:




