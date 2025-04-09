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

# [TASK 1: Sentiment Labeling & Analysis] (using NLTK VADER)
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
        
    # apply helper funct. on 'body' column
    df['sentiment'] = df['body'].astype(str).apply(get_sentiment)
    
    # map text sentiment labels to numerical scores
    score_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
    df['score'] = df['sentiment'].map(score_map)
    
    # create addit. binary column (neg_flag) to indicate negative messages
    df['neg_flag'] = df['sentiment'].apply(lambda s: 1 if s == 'NEGATIVE' else 0)
    
    # return modified DataFrame with new columns
    return df

# [TASK 2: Exploratory Data Analysis (EDA)]
def run_eda(df):
    # generates & saves visualizations from dataset
    # (countplot for sentiment distrib. & chart saved in visual. folder)
    plt.figure(figsize = (8,5))
    sns.countplot(x = 'sentiment', data = df)
    plt.title("Sentiment Distribution")
    plt.savefig("visualization/sentiment_distribution.png")
    plt.show()

# [TASK 3: Employee Score Calculation (ESC)]
def calculate_employee_scores(df):
    # calculate monthly sentiment scores per employee
    # work on copy so original df not overriden
    df_copy = df.copy()
    df_copy.set_index('date', inplace = True)
    # group data by employee & month (using 'date' column) & sum 'score'
    monthly_scores = df_copy.groupby(['employee_id', pd.Grouper(freq = 'M')])['score'].sum().reset_index()
    # returns DataFrame of monthly scores
    return monthly_scores

# [TASK 4: Employee Ranking]
def rank_employees(monthly_scores):
    # generates and prints lists of employees per month based on monthly sentiment scores
    rankings = {}
    # group by month ('date' column should have month-end timestamps)
    # for each month -> two lists:
    for month, group in monthly_scores.groupby('date'):
        # sort for pos. rank: score descending then employee id ascending
        top_positive = group.sort_values(by = ['score', 'employee_id'], ascending = [False, True]).head(3)
        # sort for pos. rank: score acsending then employee id ascending
        top_negative = group.sort_values(by = ['score', 'employee_id'], ascending = [True, True]).head(3)
        rankings[month] = {
            'positive': top_positive,
            'negative': top_negative
        }
        print(f"\nMonth: {month.date()}")
        print("Top Three Positive Employees:")
        print(top_positive[['employee_id', 'score']])
        print("Top Three Negative Employees:")
        print(top_negative[['employee_id', 'score']])
    # return dict mapping: month -> ranking
    return rankings
