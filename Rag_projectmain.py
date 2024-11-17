#!/usr/bin/env python
# coding: utf-8

# In[24]:



# Import Dependencies
#%pip install 
#%pip install feedparser
#%pip install ssl
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import openai
import numpy as np
from openai import OpenAI
import os
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



# ### Get Stock Financials and Overview

# In[25]:

ticker_symbol = input("Input ticker symbol: ")



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





# ### Get 2024 Financial Statements Only

# In[27]:

def save_crypto_data_to_pdf(ticker):
    if 'usd' in ticker.lower():
        price_data_str = price_data.to_string()
        return str(price_data_str)
    else:
        return None
   



def save_financial_data_to_string(ticker):
    # Fetch financial data using yfinance
    company = yf.Ticker(ticker)
    
    if 'usd' not in ticker.lower(): 
        income_statement = company.financials.astype(str).to_string()
        balance_sheet = company.balance_sheet.astype(str).to_string()
        cash_flow = company.cashflow.astype(str).to_string()
        company_overview = company.info
        price_data = company.history(period="2y").astype(str).to_string()
   
        


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
        complete_company_report += price_data
    else:
        price_data = company.history(period="2y").astype(str).to_string()
        return price_data
    

    return str(complete_company_report)
    
    # # Save the PDF in the folder
    # pdf_file_path = os.path.join(folder_path, f"{ticker}_financials.pdf")
    # pdf.output(pdf_file_path)

    #print(f"PDF saved successfully at: {pdf_file_path}")    

save_financial_data_to_string(ticker_symbol)


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


# 

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


# ### Extract Text from PDF

# In[18]:


# # Function to extract text from the PDF
# def extract_text_from_pdf(pdf_file):
#     with open(pdf_file, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text += page.extract_text()
#     return text


# ### Chunk text into smaller pieces

# In[19]:


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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a Hedge Fund manager offering investment price and outlook analysis and recommendations. Based on the following context: {context}, provide a complete and concise answer to the question: {question}. Ensure your response forms a full, coherent thought and does not trail off, staying within the character limit."}

        ],
        max_tokens=200
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

try: 
    answer = ask_question(question)
    print(answer)
except openai.error.RateLimitError as e:
    print(f"RateLimitError: {e}. You have exceeded your current quota. Please check your plan and billing details. Display trading volume")


# In[ ]:




