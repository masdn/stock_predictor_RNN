import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_grid_training_loss(results, save_path, cfg):
    '''
    puts four plots, one for each stock,
    in a 2x2 grid on training loss per epoch
    '''
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    hps = f"hidden={cfg['hidden_size']}  recurrent layers={cfg['num_layers']}  lr={cfg['lr']}  dropout={cfg['dropout']}"
    fig.suptitle(hps, fontsize=10, y=1.01)

    for ax, (stock, epochs, losses) in zip(axes, results):
        ax.plot(epochs, losses, label='train loss')
        ax.set_title(stock)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg MSE Loss')
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    #plt.show()
    plt.close()
    


def plot_grid_true_v_pred(results, save_path, cfg):
    '''
    puts four plots, one for each stock,
    in a 2x2 grid on true vs pred values
    '''
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    hps = f"hidden={cfg['hidden_size']}  recurrent layers={cfg['num_layers']}  lr={cfg['lr']}  dropout={cfg['dropout']}"
    fig.suptitle(hps, fontsize=10, y=1.01)

    for ax, (stock, dates, true_vals, pred_vals) in zip(axes.flatten(), results):
        ax.plot(dates, true_vals, label='True')
        ax.plot(dates, pred_vals, label='Predicted')
        ax.set_title(stock)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    #plt.show()
    plt.close()


def get_stock_data():
    '''
    gets stock data from yfinance
    '''
    stocks = ["AAPL", "NVDA", "AMZN", "WMT"]
    ds = yf.download(stocks, start="2023-04-13", end="2026-04-13")
    ds = ds.dropna()
    return ds, stocks


def preprocess(ds, stock, window_size=60):
    '''
    normalizes the data, splits into 80% training, 20%
    test, apply sliding window of size 60 to both sets
    '''
    #training only on closing prices
    close = ds['Close'][[stock]].values

    train_size = int(len(close) * 0.80)

    #normalize the data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(close[:train_size])
    test_scaled  = scaler.transform(close[train_size:])

    #split train set into windows
    X_train, y_train = [], []
    for i in range(window_size, len(train_scaled)):
        X_train.append(train_scaled[i-window_size:i, :])
        y_train.append(train_scaled[i, :])

    #split train set into windows
    test_context = np.concatenate([train_scaled[-window_size:], test_scaled])
    X_test, y_test = [], []
    for i in range(window_size, len(test_context)):
        
        X_test.append(test_context[i-window_size:i, :])
        y_test.append(test_context[i, :])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test,  y_test  = np.array(X_test),  np.array(y_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test,  dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test,  dtype=torch.float32),
        scaler
    )


def get_results_dict(mode):
    dict = {}
    if mode == 'train':
        dict = {
            'stock_name': [],
            'timestamp': [],
            'epoch': [],
            'avg_mse_loss': [],
            'fit_time': [],
        }
    elif mode == 'eval':
        dict = {
            'stock_name': None,
            'split': None,
            'timestamp': None,
            'rmse': None,
            'mape': None,
            'eval_time': None,
            'input_size': None,
            'hidden_size': None,
            'output_size': None,
            'num_layers': None,
            'dropout': None,
            'lr': None,
            'num_epochs': None,
        }
    return dict


def log(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(results, dict):
        results = [results]
    df = pd.concat([pd.DataFrame(r) for r in results if r is not None], ignore_index=True)
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    #print(df.to_string(index=False))


