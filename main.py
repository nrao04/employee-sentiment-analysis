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
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
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

# [TASK 5: Flight Risk Identification]
def identify_flight_risks(df):
    # identifies employees who have sent 4 or more neg. messages (in 30 day per.)
    # work on copy of dataFrame
    df_copy = df.copy()
    if df_copy.index.name != 'date':
        df_copy.set_index('date', inplace = True)
    # sort by employee id and date
    df_copy = df_copy.sort_values(['employee_id', 'date'])
    
    # helper funct to group by employee and calc. 30 day roll. sum on neg. messages
    def rolling_negatives(group):
        # rolling window of 30 days (using index which is datetime)
        return group['neg_flag'].rolling('30D').sum()
    
    df_copy['neg_rolling_sum'] = df_copy.groupby('employee_id', group_keys = False).apply(rolling_negatives)
    # flag employees with rolling sum >= 4 as flight risks
    flight_risk_employees = df_copy[df_copy['neg_rolling_sum'] >= 4]['employee_id'].unique().tolist()
    print("\nFlight Risk Employees: ")
    print(flight_risk_employees)
    return flight_risk_employees

# [TASK 6: Predictive Modeling]
def run_predictive_model(monthly_scores):
    # using linear regression model to predict monthly sentiment scores
    # model utilizes month num. extracted from date as feature
    monthly_scores['month_num'] = monthly_scores['date'].dt.month
    x = monthly_scores[['month_num']]
    y = monthly_scores['score']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    # train lin. regr. model & using 'fit' mthod to find best lin. rel. bet. features
    model = LinearRegression().fit(x_train, y_train)
    # using trained model t omake predictions on test dataset
    predictions = model.predict(x_test)
    # calc. mse to determine model's accuracy (lower mse -> better predictive performance)
    mse = mean_squared_error(y_test, predictions)
    # prints mean squared error of model on a test split
    print("\nPredictive Modeling:")
    print("Mean Squared Error:", mse)
    
# main 
def main():
    # load data from csv (test.csv)
    df = preprocess_and_load_data("data/test.csv")
    
    # sentiment labeling & analysis (task 1)
    df = add_sentiment_labels(df)
    
    # run exp. data analysis (task 2)
    run_eda(df)
    
    # calc. empl. monthly sent. scores (task 3)
    monthly_scores = calculate_employee_scores(df)
    
    # gener. empl. rankings based on monthly scores (task 4)
    rankings = rank_employees(monthly_scores)
    
    # identify flight risk employees (task 5)
    flight_risks = identify_flight_risks(df)
    
    # run predictive modeling (task 6)
    run_predictive_model(monthly_scores)

# main working smoothly now!
if __name__ == "__main__":
    main()