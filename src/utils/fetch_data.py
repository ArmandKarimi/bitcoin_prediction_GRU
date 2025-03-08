import yfinance as yf 
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 👈 Add the parent directory to Python's path so it access modules (e..g config) in the main directory as if it were a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NAME

def load_data(name = NAME, start = "2014-01-01", end = datetime.now()):
    df = yf.download(name , start, end)
    df.columns = df.columns.droplevel('Ticker')
    df = df.dropna()
    df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
    df = df.reset_index()

        ### Feature engineering
    df['Close_pct'] = df['Close'].pct_change()
    df['High_Low'] = df['High'] - df['Low']
    df['Open_Close'] = df['Open'] - df['Close']
    df['Open_High'] = df['Open'] - df['High']
    df['Open_Low'] = df['Open'] - df['Low']

    # When using LSTM/GRU since there is a sequence of past data 
    df["Close_1D"] = df["Close"].shift(1)
    df["Close_2D"] = df["Close"].shift(2)
    df["Close_3D"] = df["Close"].shift(3)
    df["Close_4D"] = df["Close"].shift(4)
    df["Close_5D"] = df["Close"].shift(5)
    df["Close_6D"] = df["Close"].shift(6)

    df['Volume_1D'] = df['Volume'].shift(1)
    df['Volume_2D'] = df['Volume'].shift(2)

    #moving average 3 days, one week, one month for Close
    df['MA_3D'] = df['Close'].rolling(window=3).mean()
    df['MA_7D'] = df['Close'].rolling(window=7).mean()
    df['MA_30D'] = df['Close'].rolling(window=30).mean()

    date_time = pd.to_datetime(df.pop('Date'), format='%d.%m.%Y %H:%M:%S')
    #in seconds
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    
    day = 24*60*60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    df = load_data(name = NAME)
    print(df.describe())
    #print(df.columns)