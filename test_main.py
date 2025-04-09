import os
import unittest
import pandas as pd
import textwrap
from io import StringIO
import tempfile

# import functions from main.py
from main import (
    preprocess_and_load_data,
    add_sentiment_labels,
    run_eda,
    calculate_employee_scores,
    rank_employees,
    identify_flight_risks,
    run_predictive_model
)

class TestEmployeeSentimentAnalysis(unittest.TestCase):
    
    def setUp(self):
        # create sample csv data as str
        csv_data = textwrap.dedent("""\
        employee_id,body,date
        e1,"i love my job",2023-01-01
        e2,"i hate the new policy",2023-01-02
        e3,"it is okay, nothing special",2023-01-03
        e1,"not a good day at work",2023-01-15
        e2,"absolutely fantastic experience",2023-01-20
        e3,"could be better",2023-01-25
        e1,"i hate this project",2023-01-30
        e2,"terrible service and support",2023-02-05
        e3,"i am neutral about the meeting",2023-02-10
        e1,"i love the team spirit",2023-02-15
        """)
        # load sample data from str
        self.df_sample = pd.read_csv(StringIO(csv_data))
        self.df_sample['date'] = pd.to_datetime(self.df_sample['date'])
    
    def test_preprocess_and_load_data(self):
        # test that funct. reads a csv and conv. 'date' column to datetime
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmp:
            self.df_sample.to_csv(tmp.name, index=False)
            tmp.seek(0)
            df_loaded = preprocess_and_load_data(tmp.name)
            self.assertIn('date', df_loaded.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_loaded['date']))
        os.remove(tmp.name)
    
    def test_add_sentiment_labels(self):
        # test that add_sentiment_labels adds 'sentiment', 'score', and 'neg_flag' columns
        df_labeled = add_sentiment_labels(self.df_sample.copy())
        self.assertIn('sentiment', df_labeled.columns)
        self.assertIn('score', df_labeled.columns)
        self.assertIn('neg_flag', df_labeled.columns)
        # check that all sentiment values are one of the expected strings (lower-case expected as in our main.py style)
        sentiments = set(df_labeled['sentiment'].str.lower().unique())
        self.assertTrue(sentiments.issubset({'positive', 'negative', 'neutral'}))
    
    def test_run_eda_creates_file(self):
        # ensure the 'visualization' folder exists
        if not os.path.exists("visualization"):
            os.makedirs("visualization")
        # add sentiment labels so that 'sentiment' column exists for plotting
        df_labeled = add_sentiment_labels(self.df_sample.copy())
        run_eda(df_labeled)
        self.assertTrue(os.path.exists("visualization/sentiment_distribution.png"),
                        "plot file not created in visualization folder")
        os.remove("visualization/sentiment_distribution.png")
    
    def test_calculate_employee_scores(self):
        # test that monthly employee scores are calculated and the result has required columns
        df_labeled = add_sentiment_labels(self.df_sample.copy())
        monthly_scores = calculate_employee_scores(df_labeled)
        self.assertIn('employee_id', monthly_scores.columns)
        self.assertIn('score', monthly_scores.columns)
        self.assertGreater(len(monthly_scores), 0)
    
    def test_rank_employees(self):
        # test that rank_employees returns a dict with keys 'positive' and 'negative'
        df_labeled = add_sentiment_labels(self.df_sample.copy())
        monthly_scores = calculate_employee_scores(df_labeled)
        rankings = rank_employees(monthly_scores)
        self.assertIsInstance(rankings, dict)
        self.assertGreater(len(rankings), 0)
        # check that for one month we have both positive and negative ranking dataframes
        for month, rank in rankings.items():
            self.assertIn('positive', rank)
            self.assertIn('negative', rank)
            break
    
    def test_identify_flight_risks(self):
        # test that identify_flight_risks returns a list
        df_labeled = add_sentiment_labels(self.df_sample.copy())
        risks = identify_flight_risks(df_labeled.copy())
        self.assertIsInstance(risks, list)
    
    def test_run_predictive_model(self):
        # test that run_predictive_model executes without error
        df_labeled = add_sentiment_labels(self.df_sample.copy())
        monthly_scores = calculate_employee_scores(df_labeled)
        try:
            run_predictive_model(monthly_scores)
        except Exception as e:
            self.fail("run_predictive_model() raised an exception: " + str(e))

if __name__ == '__main__':
    unittest.main()