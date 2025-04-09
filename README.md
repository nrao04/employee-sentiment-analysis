# Employee Sentiment Analysis

## Summary

This project analyzes an unlabeled dataset of employee messages to evaluate employee engagement and detect potential flight risks. The analysis includes automatic sentiment labeling, exploratory data analysis (EDA), employee scoring and ranking, flight risk identification, and predictive modeling.

---

## Top Three Positive Employees (Example: July 2011)

1. **patti.thompson@enron.com** — Score: 16
2. **sally.beck@enron.com** — Score: 9
3. **bobette.riner@ipgdirect.com** — Score: 8

## Top Three Negative Employees (Example: July 2011)

1. **lydia.delgado@enron.com** — Score: -1
2. **kayne.coulter@enron.com** — Score: 1
3. **don.baughman@enron.com** — Score: 3

> *Note: These rankings are on a monthly basis. Full monthly rankings are printed in the console output.*

---

## Employees Flagged as Flight Risks

- **bobette.riner@ipgdirect.com**
- **don.baughman@enron.com**
- **johnny.palmer@enron.com**
- **sally.beck@enron.com**

Employees are flagged if they sent 4 or more negative messages within any rolling 30-day period.

---

## Key Insights

- Sentiment distribution is fairly balanced between positive, neutral, and negative messages across the dataset.
- Certain employees consistently appear in the positive or negative rankings month over month.
- A small group of employees show frequent negative communication patterns and were identified as potential flight risks.
- The predictive model yielded a Mean Squared Error (MSE) of **11.90**, suggesting reasonable predictive power for monthly sentiment trends.

---

## Recommendations

- **Employee Engagement:**  
  Implement targeted communication strategies for employees who consistently rank negative to address underlying concerns early.

- **Monitoring Flight Risks:**  
  Closely monitor employees flagged as flight risks and engage them with proactive support, feedback sessions, or retention strategies.

- **Data-Driven Decisions:**  
  Use the predictive model’s output to forecast trends and allocate resources where morale improvements are needed most.

- **Continued Analysis:**  
  Update the analysis regularly with new communication data to maintain an accurate, up-to-date view of employee engagement and risks.

---

## Visualizations

All generated charts are saved in the `visualization` folder:
- **sentiment_distribution.png** — Bar chart showing sentiment category counts.
- **monthly_message_frequency.png** — Line chart of message volume by month.
- **monthly_average_sentiment.png** — Bar chart of average monthly sentiment scores.

---

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd employee-sentiment-analysis
