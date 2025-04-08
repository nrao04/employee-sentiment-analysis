# imports & initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import nltk

# download necessary NLTK data
nltk.download('vader_lexicon')

# preprocessing & data loading
def preprocess_and_load_data(filepath):
    # reads csv file from filepath and conv. 'date' column into datetime obj.
    df = pd.read_csv(filepath)
    df['date'] = pd.read_csv(filepath)
    return df

# sentiment labeling & analysis (using NLTK VADER)
def add_sentiment_labels(df):
    # adds sentiment labels to DataFrame using NLTK's VADER
    # labels -> based on compound score from VADER
    # >= 0.05 (POS.), <= 0.05 (NEG.), otherwise (NEUTR.)
    sid = SentimentIntensityAnalyzer()
    
    # helper funct. to deter. sentiment from text
    def get_sentiment(text):
        # use VADER to calc. polarity scores
        scores = sid.polarity_scores(text)
        # compound score rep overall sentiment
        compound = scores['compound']
        # text -> pos., neg., or netur.
        if compound >= 0.05:
            return 'POSITIVE'
        elif compound <= -0.05:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
        