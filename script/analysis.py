import gdown
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models

# Google Drive file ID for raw_analysis_ratings.csv
file_id = "1Klf46Ge8lgSedYnNomAMAdHYdk_y2GX1"
file_name = "raw_analysis_ratings.csv"

# Download the data
gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

# Load the data into a pandas DataFrame
data = pd.read_csv(file_name)

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

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(data['keywords'])
corpus = [dictionary.doc2bow(text) for text in data['keywords']]

# Perform LDA
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Display the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
