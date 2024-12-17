# Financial News Analysis

This project performs various analyses on financial news headlines, including data cleaning, sentiment analysis, and trend analysis. The goal is to extract insights from the headlines and understand their impact on the stock market.

## Table of Contents
- [Installation](#installation)
- [Data Loading](#data-loading)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Sentiment Analysis](#sentiment-analysis)
- [Keyword Extraction](#keyword-extraction)
- [Publication Trends](#publication-trends)
- [Market Events](#market-events)
- [Publication Time Analysis](#publication-time-analysis)
- [Publisher Analysis](#publisher-analysis)
- [Stock Analysis](#stock-analysis)
- [Correlation Analysis](#correlation-analysis)

## Installation

To run this project, you need to install the following Python libraries:

```bash
pip install gdown pandas matplotlib textblob sklearn gensim talib

Data Loading
Load the financial news data from a CSV file.

Data Cleaning
Get the number of rows.

Rename the unnamed column to "SNo".

Check for missing values.

Check for duplicate rows.

Convert date column to datetime with ISO8601 format.

Exploratory Data Analysis
Calculate basic statistics for headline lengths.

Count the number of articles per publisher.

Analyze publication dates.

Plot publication trends over time.

Analyze publication trends by day of the week.

Analyze publication trends by month.

Analyze publication trends by year.

Sentiment Analysis
Normalize text data by converting to lowercase.

Perform sentiment analysis using TextBlob.

Classify sentiment as positive, negative, or neutral.

Display and plot sentiment distribution.

Keyword Extraction
Extract common keywords or phrases using CountVectorizer.

Extract common phrases (bigrams).

Extract common phrases (trigrams).

Publication Trends
Analyze publication dates.

Plot publication trends over time.

Market Events
List significant market events.

Plot publication trends over time with market events.

Publication Time Analysis
Extract hour from date.

Convert hour to AM/PM format.

Analyze and plot publication trends by hour in AM/PM format.

Publisher Analysis
Count the number of articles per publisher.

Plot the top publishers.

Analyze the type of news reported by top publishers.

Extract domains from publisher email addresses.

Count the number of articles per domain.

Plot the top domains.

Stock Analysis
Combine stock price data from multiple CSV files.

Normalize dates in stock data.

Calculate daily stock returns.

Analyze and visualize stock data with indicators (SMA, RSI, MACD).

Plot combined indicators for all stocks.

Calculate and plot daily returns for all stocks.

Correlation Analysis
Perform sentiment analysis on news headlines.

Aggregate daily sentiment scores.

Merge sentiment data with stock price data.

Drop rows with missing values.

Calculate the Pearson correlation coefficient between average daily sentiment scores and stock daily returns.

Calculate correlation for positive and negative sentiments.

Display and plot correlation results.