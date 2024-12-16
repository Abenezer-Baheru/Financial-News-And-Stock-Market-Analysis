import gdown
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models

# Load the data 
file_path = "../src/data/raw_analysis_ratings.csv" 
data = pd.read_csv(file_path)

# Get the number of rows
num_rows = data.shape[0]
print(f"The number of rows in the DataFrame is: {num_rows}")

# Rename the unnamed column to "SNo"
data.rename(columns={data.columns[0]: 'SNo'}, inplace=True)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicates = data.duplicated().sum()
print("Duplicate Rows:\n", duplicates)

# Convert date column to datetime with ISO8601 format
data['date'] = pd.to_datetime(data['date'], format='ISO8601')

# Calculate basic statistics for headline lengths
data['headline_length'] = data['headline'].apply(len)
headline_stats = data['headline_length'].describe()
print("Headline Length Statistics:\n", headline_stats)

# Count the number of articles per publisher
articles_per_publisher = data['publisher'].value_counts()
print("\nArticles per Publisher:\n", articles_per_publisher)

# Analyze publication dates
publication_trends = data['date'].dt.date.value_counts().sort_index()
print("\nPublication Trends Over Time:\n", publication_trends)

# Plot publication trends over time
publication_trends.plot(kind='line', figsize=(14, 8))
plt.title('Publication Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()

# Extract day of the week from date
data['day_of_week'] = data['date'].dt.day_name()

# Analyze publication trends by day of the week
publication_by_day = data['day_of_week'].value_counts().sort_index()

# Sort in ascending order
publication_by_day_sorted = publication_by_day.sort_values()

# Plot publication trends by day of the week in ascending order
publication_by_day_sorted.plot(kind='bar', figsize=(10, 6))
plt.title('Publication Trends by Day of the Week (Ascending Order)')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Articles')
plt.show()

# Extract month from date
data['month'] = data['date'].dt.month_name()

# Analyze publication trends by month
publication_by_month = data['month'].value_counts().sort_index()

# Sort in ascending order
publication_by_month_sorted = publication_by_month.sort_values()

# Plot publication trends by month in ascending order
publication_by_month_sorted.plot(kind='bar', figsize=(10, 6))
plt.title('Publication Trends by Month (Ascending Order)')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
plt.show()

# Extract year from date
data['year'] = data['date'].dt.year

# Analyze publication trends by year
publication_by_year = data['year'].value_counts().sort_index()

# Sort in ascending order
publication_by_year_sorted = publication_by_year.sort_values()

# Plot publication trends by year in ascending order
publication_by_year_sorted.plot(kind='bar', figsize=(10, 6))
plt.title('Publication Trends by Year (Ascending Order)')
plt.xlabel('Year')
plt.ylabel('Number of Articles')
plt.show()

# Normalize text data by converting to lowercase
data['headline'] = data['headline'].str.lower()

# Perform sentiment analysis
data['sentiment'] = data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Classify sentiment as positive, negative, or neutral
data['sentiment_label'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Display sentiment distribution
sentiment_distribution = data['sentiment_label'].value_counts()
print("Sentiment Distribution:\n", sentiment_distribution)

# Plot sentiment distribution
sentiment_distribution.plot(kind='bar', figsize=(10, 6))
plt.title('Sentiment Distribution of Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Number of Articles')
plt.show()

# Extract common keywords or phrases using CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20)
X_count = count_vectorizer.fit_transform(data['headline'])
common_phrases = count_vectorizer.get_feature_names_out()

# Count common phrases
phrase_counts = X_count.toarray().sum(axis=0)
phrase_counts_dict = dict(zip(common_phrases, phrase_counts))

# Sort common phrases in descending order
sorted_phrase_counts = dict(sorted(phrase_counts_dict.items(), key=lambda item: item[1], reverse=True))
print("Top 20 Common Phrases (Descending Order):\n", sorted_phrase_counts)

# Plot top 20 common phrases in descending order
plt.figure(figsize=(12, 8))
plt.bar(sorted_phrase_counts.keys(), sorted_phrase_counts.values())
plt.title('Top 20 Common Phrases in Headlines (Descending Order)')
plt.xlabel('Phrases')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# Extract common phrases (bigrams) using CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=20)
X_count = count_vectorizer.fit_transform(data['headline'])
common_phrases = count_vectorizer.get_feature_names_out()

# Count common phrases
phrase_counts = X_count.toarray().sum(axis=0)
phrase_counts_dict = dict(zip(common_phrases, phrase_counts))

# Sort common phrases in descending order
sorted_phrase_counts = dict(sorted(phrase_counts_dict.items(), key=lambda item: item[1], reverse=True))
print("Top 20 Common Phrases (Descending Order):\n", sorted_phrase_counts)

# Plot top 20 common phrases in descending order
plt.figure(figsize=(12, 8))
plt.bar(sorted_phrase_counts.keys(), sorted_phrase_counts.values())
plt.title('Top 20 Common Phrases in Headlines (Descending Order)')
plt.xlabel('Phrases')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# Extract common phrases (trigrams) using CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(3, 3), max_features=20)
X_count = count_vectorizer.fit_transform(data['headline'])
common_phrases = count_vectorizer.get_feature_names_out()

# Count common phrases
phrase_counts = X_count.toarray().sum(axis=0)
phrase_counts_dict = dict(zip(common_phrases, phrase_counts))

# Sort common phrases in descending order
sorted_phrase_counts = dict(sorted(phrase_counts_dict.items(), key=lambda item: item[1], reverse=True))
print("Top 20 Common Phrases (Descending Order):\n", sorted_phrase_counts)

# Plot top 20 common phrases in descending order
plt.figure(figsize=(12, 8))
plt.bar(sorted_phrase_counts.keys(), sorted_phrase_counts.values())
plt.title('Top 20 Common Phrases in Headlines (Descending Order)')
plt.xlabel('Phrases')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# Analyze publication dates
publication_trends = data['date'].dt.date.value_counts().sort_index()

# Plot publication trends over time
plt.figure(figsize=(14, 8))
publication_trends.plot(kind='line')
plt.title('Publication Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()

# List of significant market events
market_events = {
    '2009-03-09': 'Recovery from Global Financial Crisis',
    '2010-05-09': 'European Sovereign Debt Crisis',
    '2011-08-02': 'US Debt Ceiling Crisis',
    '2011-12-17': 'Arab Spring',
    '2013-10-01': 'US Government Shutdown',
    '2014-06-20': 'Oil Price Crash',
    '2015-06-12': 'Chinese Stock Market Crash',
    '2016-06-23': 'Brexit Vote',
    '2016-11-08': 'US Presidential Election',
    '2018-07-06': 'US-China Trade War',
    '2020-03-11': 'COVID-19 Pandemic'
}

# List of colors for the events
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

# Plot publication trends over time with market events
plt.figure(figsize=(14, 8))
publication_trends.plot(kind='line')
for i, (event_date, event_name) in enumerate(market_events.items()):
    plt.axvline(pd.to_datetime(event_date), color=colors[i % len(colors)], linestyle='--', label=event_name)
plt.title('Publication Trends Over Time with Market Events')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.legend()
plt.show()

# Extract hour from date
data['hour'] = data['date'].dt.hour

# Convert hour to AM/PM format
def convert_to_ampm(hour):
    if hour == 0:
        return '12 AM'
    elif hour < 12:
        return f'{hour} AM'
    elif hour == 12:
        return '12 PM'
    else:
        return f'{hour - 12} PM'

data['hour_ampm'] = data['hour'].apply(convert_to_ampm)

# Analyze publication trends by hour in AM/PM format
publication_by_hour_ampm = data['hour_ampm'].value_counts().sort_values(ascending=False)

# Print publication trends by hour in AM/PM format in descending order
print("Publication Trends by Hour (AM/PM Format) in Descending Order:")
for hour_ampm, count in publication_by_hour_ampm.items():
    print(f"{hour_ampm}: {count} articles")

# Plot publication trends by hour in AM/PM format
plt.figure(figsize=(10, 6))
publication_by_hour_ampm.plot(kind='bar')
plt.title('Publication Trends by Hour (AM/PM Format)')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.show()

# Perform sentiment analysis
data['headline'] = data['headline'].str.lower()
data['sentiment'] = data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['sentiment_label'] = data['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Count the number of articles per publisher
articles_per_publisher = data['publisher'].value_counts()

# Print the top publishers
print("Top Publishers by Number of Articles:")
print(articles_per_publisher.head(10))

# Plot the top publishers
plt.figure(figsize=(10, 6))
articles_per_publisher.head(10).plot(kind='bar')
plt.title('Top 10 Publishers by Number of Articles')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.show()

# Analyze the type of news reported by top publishers
top_publishers = articles_per_publisher.head(10).index
sentiment_data = pd.DataFrame()

for publisher in top_publishers:
    publisher_data = data[data['publisher'] == publisher]
    sentiment_distribution = publisher_data['sentiment_label'].value_counts(normalize=True)
    sentiment_data[publisher] = sentiment_distribution

# Transpose the DataFrame for plotting
sentiment_data = sentiment_data.T
sentiment_data = sentiment_data.fillna(0)

# Plot the combined sentiment distribution
sentiment_data.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Sentiment Distribution for Top 10 Publishers')
plt.xlabel('Publisher')
plt.ylabel('Proportion of Articles')
plt.legend(title='Sentiment')
plt.xticks(rotation=45)
plt.show()

# Print the sentiment distribution table
print("Sentiment Distribution for Top 10 Publishers:")
print(sentiment_data)

# Function to extract domain from email address
def extract_domain(email):
    match = re.search(r'@([\w.-]+)', email)
    return match.group(1) if match else None

# Extract domains from publisher email addresses
data['publisher_domain'] = data['publisher'].apply(extract_domain)

# Count the number of articles per domain
articles_per_domain = data['publisher_domain'].value_counts()

# Print the top domains
print("Top Domains by Number of Articles:")
print(articles_per_domain.head(10))

# Plot the top domains
plt.figure(figsize=(10, 6))
articles_per_domain.head(10).plot(kind='bar')
plt.title('Top 10 Domains by Number of Articles')
plt.xlabel('Domain')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.show()