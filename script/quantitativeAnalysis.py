import pandas as pd
import talib
import matplotlib.pyplot as plt
import os

# List of file paths
file_paths = [
    "../src/data/AAPL_historical_data.csv",
    "../src/data/AMZN_historical_data.csv",
    "../src/data/GOOG_historical_data.csv",
    "../src/data/META_historical_data.csv",
    "../src/data/MSFT_historical_data.csv",
    "../src/data/NVDA_historical_data.csv",
    "../src/data/TSLA_historical_data.csv"
]

# Step 1: Combine the data
combined_data = pd.DataFrame()

for file_path in file_paths:
    data = pd.read_csv(file_path)
    stock_symbol = os.path.basename(file_path).split('_')[0]
    data['Stock'] = stock_symbol
    combined_data = pd.concat([combined_data, data], ignore_index=True)

print(combined_data.head())
combined_data.to_csv("../src/data/combined_historical_data.csv", index=False)
print(combined_data.tail())

# Step 2: Analyze each stock and visualize data
def analyze_stock(file_path):
    data = pd.read_csv(file_path)
    stock_symbol = os.path.basename(file_path).split('_')[0]
    data['Date'] = pd.to_datetime(data['Date'])
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.plot(data['Date'], data['SMA_50'], label='50-Day SMA')
    plt.plot(data['Date'], data['SMA_200'], label='200-Day SMA')
    plt.title(f'{stock_symbol} Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['RSI'], label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'{stock_symbol} RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['MACD'], label='MACD')
    plt.plot(data['Date'], data['MACD_Signal'], label='MACD Signal')
    plt.bar(data['Date'], data['MACD_Hist'], label='MACD Histogram')
    plt.title(f'{stock_symbol} MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.show()

for file_path in file_paths:
    analyze_stock(file_path)

# Step 3: Plot combined indicators for all stocks
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
stock_data = {}

for file_path, color in zip(file_paths, colors):
    data = pd.read_csv(file_path)
    stock_symbol = os.path.basename(file_path).split('_')[0]
    data['Date'] = pd.to_datetime(data['Date'])
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data[stock_symbol] = (data, color)

plt.figure(figsize=(14, 8))
for stock_symbol, (data, color) in stock_data.items():
    plt.plot(data['Date'], data['SMA_50'], label=f'{stock_symbol} 50-Day SMA', color=color)
plt.title('50-Day Moving Averages for All Stocks')
plt.xlabel('Date')
plt.ylabel('50-Day SMA')
plt.legend()
plt.show()

plt.figure(figsize=(14, 8))
for stock_symbol, (data, color) in stock_data.items():
    plt.plot(data['Date'], data['RSI'], label=f'{stock_symbol} RSI', color=color)
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('RSI for All Stocks')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

plt.figure(figsize=(14, 8))
for stock_symbol, (data, color) in stock_data.items():
    plt.plot(data['Date'], data['MACD'], label=f'{stock_symbol} MACD', color=color)
plt.title('MACD for All Stocks')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
plt.show()

# Step 4: Calculate and plot daily returns for all stocks
for file_path, color in zip(file_paths, colors):
    data = pd.read_csv(file_path)
    stock_symbol = os.path.basename(file_path).split('_')[0]
    data['Date'] = pd.to_datetime(data['Date'])
    data['Daily_Return'] = data['Close'].pct_change()
    stock_data[stock_symbol] = (data, color)

plt.figure(figsize=(14, 8))
for stock_symbol, (data, color) in stock_data.items():
    plt.plot(data['Date'], data['Daily_Return'], label=f'{stock_symbol} Daily Return', color=color)
plt.title('Daily Returns for All Stocks')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()