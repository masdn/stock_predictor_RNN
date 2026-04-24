import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time
import utils
from utils import get_stock_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dirpath = './logs/base_LSTM'


class base_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers, dropout, lr, num_epochs, name):
        super(base_LSTM, self).__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.mse_loss = nn.MSELoss()

        lstm_dropout = dropout if num_layers > 1 else 0
        self.model = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=lstm_dropout,
                             batch_first=True)

        self.out_layer = nn.Linear(hidden_size, output_size)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, input):
        output, (h_n, c_n) = self.model(input)
        final = self.out_layer(output[:, -1, :])
        return final


    def mape(self, preds, targets):
        return float(torch.mean(torch.abs((preds - targets) / targets)).item() * 100)


    def fit(self, train_loader, stock_name=''):
        self.train(mode=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_results = utils.get_results_dict('train')

        start = time.time()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = self(X_batch)
                loss = self.mse_loss(output, y_batch)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()

            fit_time = time.time() - start
            avg_loss = epoch_loss / len(train_loader)

            train_results['avg_mse_loss'].append(avg_loss)
            train_results['stock_name'].append(stock_name)
            train_results['timestamp'].append(timestamp)
            train_results['epoch'].append(epoch)
            train_results['fit_time'].append(fit_time)

        total_fit_time = time.time() - start
        utils.log(train_results, f'{dirpath}/{self.name}/all_{self.name}_train_results.csv')

        return total_fit_time, train_results


    def evaluate(self, test_loader, scaler=None, stock_name='', split='test'):
        self.eval()
        all_preds = []
        all_targets = []

        start = time.time()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = self(X_batch)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch)
        eval_time = time.time() - start

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if scaler is not None:
            all_preds = torch.tensor(scaler.inverse_transform(all_preds.numpy()))
            all_targets = torch.tensor(scaler.inverse_transform(all_targets.numpy()))

        mse = self.mse_loss(all_preds, all_targets).item()
        rmse = mse ** 0.5
        mape = self.mape(all_preds, all_targets)

        eval_results = utils.get_results_dict('eval')
        eval_results['stock_name'] = [stock_name]
        eval_results['split'] = [split]
        eval_results['timestamp'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        eval_results['rmse'] = [rmse]
        eval_results['mape'] = [mape]
        eval_results['eval_time'] = [eval_time]
        eval_results['input_size'] = [self.input_size]
        eval_results['hidden_size'] = [self.hidden_size]
        eval_results['output_size'] = [self.output_size]
        eval_results['num_layers'] = [self.num_layers]
        eval_results['dropout'] = [self.dropout]
        eval_results['lr'] = [self.lr]
        eval_results['num_epochs'] = [self.num_epochs]
        utils.log(eval_results, f'{dirpath}/{self.name}/eval_summary/eval_results.csv')

        self.train()
        return rmse, mape, all_preds, all_targets


def train_test_eval(cfg):
    print(f"\n{'-'*40} Starting... {'-'*40}")
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds, stocks = get_stock_data()
    model_type = dirpath.split('_')[-1]
    test_dates = ds.index[int(len(ds) * 0.80):]

    loss_results = []
    pred_results = []

    for stock in stocks:
        X_train, X_test, y_train, y_test, scaler = utils.preprocess(ds, stock)
        print(f"\n{'-'*40} Training: {stock} {'-'*40}")

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=32, shuffle=False)

        model = base_LSTM(
            input_size=cfg['input_size'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            lr=cfg['lr'],
            num_epochs=cfg['num_epochs'],
            name=cfg['name'],
        ).to(device)

        train_time, train_log = model.fit(train_loader, stock_name=stock)
        loss_results.append((stock, train_log['epoch'], train_log['avg_mse_loss']))

        os.makedirs(f'./models/base_LSTM/{stock}', exist_ok=True)
        torch.save(model.state_dict(), f'./models/base_LSTM/{stock}/{stock}_{cfg["name"]}.pt')

        train_rmse, train_mape, _, _           = model.evaluate(train_loader, scaler=scaler, stock_name=stock, split='train')
        test_rmse,  test_mape,  preds, targets = model.evaluate(test_loader,  scaler=scaler, stock_name=stock, split='test')
        pred_results.append((stock, test_dates, targets.numpy().flatten(), preds.numpy().flatten()))

        summary = {
            'stock_name':  [stock],
            'train_rmse':  [round(train_rmse, 2)],
            'test_rmse':   [round(test_rmse,  2)],
            'train_mape':  [round(train_mape, 2)],
            'test_mape':   [round(test_mape,  2)],
            'fit_time':    [round(train_time, 1)],
            'name':        [cfg['name']],
            'hidden_size': [cfg['hidden_size']],
            'num_layers':  [cfg['num_layers']],
            'dropout':     [cfg['dropout']],
            'lr':          [cfg['lr']],
            'num_epochs':  [cfg['num_epochs']],
        }
        utils.log(summary, f'{dirpath}/{cfg["name"]}/{stock}/summary_results.csv')
        print(f'{stock} | Train RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f} | Train MAPE: {train_mape:.2f}% | Test MAPE: {test_mape:.2f}% | Train time: {train_time:.1f}s')

    stocks_str = '_'.join([r[0] for r in pred_results])
    utils.plot_grid_training_loss(loss_results,
        save_path=f'{dirpath}/{cfg["name"]}/{model_type}_{cfg["name"]}_{stocks_str}_train_loss.png')
    utils.plot_grid_true_v_pred(pred_results,
        save_path=f'{dirpath}/{cfg["name"]}/{model_type}_{cfg["name"]}_{stocks_str}_true_vs_pred.png')

    end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Began: {start_timestamp} | Finished: {end_timestamp}')
    print(f"\n{'-'*40} All 4 Stocks Done {'-'*40}")


if __name__ == '__main__':
    cfg = {
        'input_size':  1,
        'hidden_size': 64,
        'output_size': 1,
        'num_layers':  2,
        'dropout':     0.2,
        'lr':          0.001,
        'num_epochs':  100,
        'name':        'generic_stacked',
    }
    train_test_eval(cfg)
