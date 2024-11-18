import sys
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
from Rag_projectmain import answer, ticker_symbol, question 


# Add the absolute path where config.py is located
sys.path.append('/Users/qtjefferies/4Geeks-RAG-Example/RAG_Files/Main.py')

# Import the variable from config.py

# Print the imported variable
# stock = get_stock_rag(ticker_symbol)
print(question)
print(answer)

