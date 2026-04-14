import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

ds = yf.download(["AAPL", "NVDA", "AMZN", "WMT"], start="2023-04-13", end="2026-04-13")

ds = ds.dropna()

#each row is a new training day
#5 price fields
#4 tickers
# AAPL, NVDA, AMZN, WMT
# Open - open price
# High - max price in t.day
# Low - min price in t.day
# Volume - number of shares traded
# Close - price at end of t.day (target/labels)
print(ds.shape)
print(ds.head())
ds_np = np.array(ds)


# normalize labels (just like in the paper)
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(ds[['Close']])
window_size = 60
X, y = []
for i in range(window_size, len(scaled_prices)):
    X.append(np.column_stack((
        scaled_prices[i-window_size:i,0],
        )))
    y.append(scaled_prices[i, 0])
X,y = np.array(X), np.array(y)

train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


model = nn.Sequential(

        nn.RNN(input_size=1,
               num_layers=1,
               hidden_size=64),
        )
        #nn.Dropout(0.2))



# TODO get the original labels back 
#original_labels = scaler.inverse_transform(predictions)

