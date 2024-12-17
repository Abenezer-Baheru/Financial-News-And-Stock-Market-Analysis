import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime

# Load Financial News Data
news_data = pd.read_csv("../src/data/raw_analysis_ratings.csv")

# Load Stock Price Data
file_paths = [
    "../src/data/AAPL_historical_data.csv",
    "../src/data/AMZN_historical_data.csv",
    "../src/data/GOOG_historical_data.csv",
    "../src/data/META_historical_data.csv",
    "../src/data/MSFT_historical_data.csv",
    "../src/data/NVDA_historical_data.csv",
    "../src/data/TSLA_historical_data.csv"
]

# Combine Stock Price Data
combined_stock_data = pd.DataFrame()
for file_path in file_paths:
    data = pd.read_csv(file_path)
    stock_symbol = os.path.basename(file_path).split('_')[0]
    data['Stock'] = stock_symbol
    combined_stock_data = pd.concat([combined_stock_data, data], ignore_index=True)

# Normalize Dates
news_data['date'] = pd.to_datetime(news_data['date'], format='ISO8601').dt.date
combined_stock_data['Date'] = pd.to_datetime(combined_stock_data['Date']).dt.date

# Perform Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

news_data['Sentiment'] = news_data['headline'].apply(get_sentiment)

# Calculate Daily Stock Returns
combined_stock_data['Daily_Return'] = combined_stock_data.groupby('Stock')['Close'].pct_change()

# Aggregate Sentiments
daily_sentiment = news_data.groupby(['date', 'stock'])['Sentiment'].mean().reset_index()

# Merge Sentiment with Stock Data
merged_data = pd.merge(combined_stock_data, daily_sentiment, left_on=['Date', 'Stock'], right_on=['date', 'stock'], how='left')

# Drop rows with missing values
merged_data.dropna(subset=['Daily_Return', 'Sentiment'], inplace=True)

# Calculate Overall Correlation
overall_correlation = merged_data.groupby('Stock').apply(lambda x: x['Daily_Return'].corr(x['Sentiment']))

# Filter Positive and Negative Sentiments
positive_sentiment = news_data[news_data['Sentiment'] > 0]
negative_sentiment = news_data[news_data['Sentiment'] < 0]

# Aggregate Daily Sentiments for Positive and Negative
positive_daily_sentiment = positive_sentiment.groupby(['date', 'stock'])['Sentiment'].mean().reset_index()
negative_daily_sentiment = negative_sentiment.groupby(['date', 'stock'])['Sentiment'].mean().reset_index()

# Merge Positive and Negative Sentiments with Stock Data
positive_merged_data = pd.merge(combined_stock_data, positive_daily_sentiment, left_on=['Date', 'Stock'], right_on=['date', 'stock'], how='left')
negative_merged_data = pd.merge(combined_stock_data, negative_daily_sentiment, left_on=['Date', 'Stock'], right_on=['date', 'stock'], how='left')

# Drop rows with missing values
positive_merged_data.dropna(subset=['Daily_Return', 'Sentiment'], inplace=True)
negative_merged_data.dropna(subset=['Daily_Return', 'Sentiment'], inplace=True)

# Calculate Correlation for Positive and Negative Sentiments
positive_correlation = positive_merged_data.groupby('Stock').apply(lambda x: x['Daily_Return'].corr(x['Sentiment']))
negative_correlation = negative_merged_data.groupby('Stock').apply(lambda x: x['Daily_Return'].corr(x['Sentiment']))

# Display Correlation
print("Overall Sentiment Correlation with Stock Returns:")
print(overall_correlation)

print("\nPositive Sentiment Correlation with Stock Returns:")
print(positive_correlation)

print("\nNegative Sentiment Correlation with Stock Returns:")
print(negative_correlation)

# Plot Correlation
plt.figure(figsize=(10, 6))
overall_correlation.plot(kind='bar', color='blue', label='Overall')
positive_correlation.plot(kind='bar', color='green', label='Positive', alpha=0.7)
negative_correlation.plot(kind='bar', color='red', label='Negative', alpha=0.7)
plt.title('Correlation between Daily Returns and Sentiment')
plt.xlabel('Stock')
plt.ylabel('Correlation Coefficient')
plt.legend()
plt.show()