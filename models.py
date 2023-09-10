import numpy as np
import pandas as pd
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy as dc
import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error as mape


def std_model(data, threshold: int = 3, roll: bool = False):
    """

    detect anomalies based on threshold*sigma rule
    :param data: data
    :param threshold: std threshold (usual from 2 to 5)
    :param roll: whether rolling model or not
    :return: anomalies : dataframe,
    bounds: namedtuple(high, low)

    """

    if roll:

        anomalies = np.zeros(len(data), dtype='bool')

        high = np.zeros(len(data), dtype='float')
        low = np.zeros(len(data), dtype='float')

        for key, _ in enumerate(data):

            mean = data[:key].mean()
            std = data[:key].std()

            high[key] = mean + threshold * std
            low[key] = mean - threshold * std

            if data.iloc[key].values > high[key] or data.iloc[key].values < low[key]:
                anomalies[key] = True

        ntup = namedtuple('Bounds', ['high', 'low'])

        bounds = ntup(high, low)

        return pd.Series(anomalies), bounds

    else:

        mean = data.mean()
        std = data.std()

        high = mean + threshold * std
        low = mean - threshold * std

        boarders = namedtuple('Bounds', ['high', 'low'])
        bounds = boarders(high, low)

        anomalies = pd.concat([data > high, data < low], axis=1).any(axis=1)

        return pd.Series(anomalies), bounds


def iqr_model(data, threshold=3, roll: bool = False):
    """

    inter quartile range model
    :param data:data
    :param threshold:model threshold
    :param roll: True/False whether rolling model or not
    :return:anomalies, bounds

    """

    if roll:

        anomalies = np.zeros(len(data), dtype='bool')

        high = np.zeros(len(data), dtype='float')
        low = np.zeros(len(data), dtype='float')

        for key, _ in enumerate(data):

            iqr = data[:key].quantile(0.75) - data[:key].quantile(0.25)

            high[key] = data[:key].quantile(0.75) + (iqr * threshold)

            low[key] = data[:key].quantile(0.25) - (iqr * threshold)

            if data.iloc[key].values > high[key] or data.iloc[key].values < low[key]:
                anomalies[key] = True

        ntup = namedtuple('Bounds', ['high', 'low'])

        bounds = ntup(high, low)

        return pd.Series(anomalies), bounds

    else:

        iqr = data.quantile(0.75) - data.quantile(0.25)

        high = data.quantile(0.75) + (iqr * threshold)
        low = data.quantile(0.25) - (iqr * threshold)

        ntup = namedtuple('Bounds', ['high', 'low'])

        bounds = ntup(high, low)

        anomalies = pd.concat([data > high, data < low], axis=1).any(axis=1)

        return pd.Series(anomalies), bounds


class ModelLSTM(nn.Module):
    """
    This class contains __init__ and forward functions of LSTM model for next anomaly detection
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(ModelLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])

        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DataPrep(TimeSeriesDataset):

    """
    This class prepares data for push it to LSTM model
    """

    def __init__(self, data, split_index: int, n_steps: int = 5):
        """
        Initialize
        :param data: Time series
        :param split_index: index which splits dataset to train and test
        :param n_steps: Look back parameter
        """
        super(TimeSeriesDataset, self).__init__()

        self.data = data

        self.n_steps = n_steps

        self.split_index = split_index

        shifted_df = utils.prepare_dataframe_for_lstm(data, n_steps)
        shifted_df_as_np = shifted_df.to_numpy()

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        shifted_df_as_np = self.scaler.fit_transform(shifted_df_as_np)

        X = shifted_df_as_np[:, 1:]
        y = shifted_df_as_np[:, 0]

        self.X = dc(np.flip(X, axis=1))

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        X_train = X_train.reshape((-1, n_steps, 1))
        X_test = X_test.reshape((-1, n_steps, 1))

        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

        self.X_train = torch.tensor(X_train).float()
        self.y_train = torch.tensor(y_train).float()

        self.X_test = torch.tensor(X_test).float()
        self.y_test = torch.tensor(y_test).float()

        self.train_dataset = TimeSeriesDataset(X_train, y_train)
        self.test_dataset = TimeSeriesDataset(X_test, y_test)

        self.batch_size = 16

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


# dataset = pd.read_csv('data/Data.csv', sep=';')
# df = dataset[['Time', 'x013']]
# model = ModelLSTM(1, 8, 4)


class TrainModel(nn.Module, DataPrep):

    """
    This class contains train and val functions one epoch and general train, test on time series split list
    """

    @staticmethod
    def train_one_epoch(nn_model,
                        train_loader,
                        loss_function=nn.MSELoss()):
        """
        One epoch train function
        :param nn_model: model
        :param train_loader: loader from DataPrep class
        :param loss_function: loss function (default MSELoss)
        :return:
        """
        nn_model.train(True)
        learning_rate = 0.001
        optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].float(), batch[1].float()

            output = nn_model(x_batch.float())
            loss = loss_function(output, y_batch.float())
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

    @staticmethod
    def validate_one_epoch(nn_model,
                           test_loader,
                           loss_function=nn.MSELoss()):
        """
        Validate one epoch function
        :param nn_model: model
        :param test_loader: test loader from class DataPrep
        :param loss_function: loss function (default MSELoss)
        :return:
        """
        nn_model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].float(), batch[1].float()

            with torch.no_grad():
                output = nn_model(x_batch.float())
                loss = loss_function(output, y_batch.float())
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()

    def train_model(self,
                    nn_model,
                    data,
                    num_epochs: int = 5,
                    n_splits: int = 5,
                    loss_function=nn.MSELoss(),
                    plot: bool = False,
                    all_outputs: bool = False):

        """
        Training model function via time series split
        :param nn_model: model
        :param data: time series
        :param num_epochs: int: num epochs (default = 5)
        :param n_splits: int: num splits of time series (default = 5)
        :param loss_function: loss function (default = MSELoss)
        :param plot: bool:  Need plot residuals or not (default False)
        :param all_outputs: bool: if you need list of actual/predict data on all time series splits (default False)
        :return:
        actual_train: list[list of values] or list[-1] of values whether all_outputs param,
        predict_train: list[list of values] or list[-1] of values whether all_outputs param,
        actual_test: list[list of values] or list[-1] of values whether all_outputs param,
        predict_tests: list[list of values] or list[-1] of values whether all_outputs param,
        train_losses: list of train losses,
        test_losses: list of test losses
        """

        tscv = TimeSeriesSplit(n_splits)

        dd = []

        train_losses = []
        test_losses = []

        act_trains = []
        pred_trains = []

        act_tests = []
        pred_tests = []

        for i, (train_index, test_index) in enumerate(tscv.split(DataPrep(data, split_index=len(data)).X)):

            dd.append(DataPrep(data[:test_index[-1]], train_index[-1]))

            for epoch in range(num_epochs):
                print(f'Epoch: {epoch + 1}')
                self.train_one_epoch(nn_model, dd[i].train_loader, loss_function)
                self.validate_one_epoch(nn_model, dd[i].test_loader, loss_function)

            with torch.no_grad():

                predicted = nn_model(dd[i].X_train).numpy()

            train_predictions = predicted.flatten()

            dummies = np.zeros((dd[i].X_train.shape[0], dd[i].n_steps + 1))
            dummies[:, 0] = train_predictions
            dummies = dd[i].scaler.inverse_transform(dummies)

            train_predictions = dc(dummies[:, 0])
            pred_trains.append(train_predictions)

            dummies = np.zeros((dd[i].X_train.shape[0], dd[i].n_steps + 1))
            dummies[:, 0] = dd[i].y_train.flatten()
            dummies = dd[i].scaler.inverse_transform(dummies)

            new_y_train = dc(dummies[:, 0])
            act_trains.append(new_y_train)

            test_predictions = nn_model(dd[i].X_test).detach().numpy().flatten()

            dummies = np.zeros((dd[i].X_test.shape[0], dd[i].n_steps + 1))
            dummies[:, 0] = test_predictions
            dummies = dd[i].scaler.inverse_transform(dummies)

            test_predictions = dc(dummies[:, 0])
            pred_tests.append(test_predictions)

            dummies = np.zeros((dd[i].X_test.shape[0], dd[i].n_steps + 1))
            dummies[:, 0] = dd[i].y_test.flatten()
            dummies = dd[i].scaler.inverse_transform(dummies)

            new_y_test = dc(dummies[:, 0])
            act_tests.append(new_y_test)

            if plot:
                plt.plot(dd[i].y_train, label='Actual')
                plt.plot(predicted, label='Predicted')
                plt.title(f'split {i}')
                plt.xlabel('Day')
                plt.ylabel('Y')
                plt.legend()
                plt.show()

                plt.plot(new_y_test, label='Actual')
                plt.plot(test_predictions, label='Predicted')
                plt.title(f'split {i}')
                plt.xlabel('Day')
                plt.ylabel('Y')
                plt.legend()
                plt.show()

            train_losses.append(mape(new_y_train, train_predictions) * 100)
            test_losses.append(mape(new_y_test, test_predictions) * 100)

            print(f'Train MAPE: {mape(new_y_train, train_predictions)},'
                  f'Test MAPE: {mape(new_y_test, test_predictions)}')

        if all_outputs:

            act_trains, pred_trains, act_tests, pred_tests = \
                act_trains[-1], pred_trains[-1], act_tests[-1], pred_tests[-1]
            return act_trains, pred_trains, act_tests, pred_tests, train_losses, test_losses

        else:

            return act_trains, pred_trains, act_tests, pred_tests, train_losses, test_losses

    def __init__(self,
                 nn_model,
                 data,
                 num_epochs: int = 5,
                 n_splits: int = 5,
                 loss_function=nn.MSELoss(),
                 plot: bool = False,
                 all_outputs: bool = False):
        super().__init__()

        self.nn_model = nn_model

        self.data = data

        self.num_epochs = num_epochs

        self.n_splits = n_splits

        self.loss_function = loss_function

        self.plot = plot

        self.all_outputs = all_outputs

        self.new_y_train, \
        self.train_predictions, \
        self.new_y_test, \
        self.test_predictions, \
        self.train_losses, \
        self.test_losses = self.train_model(nn_model, data, num_epochs, n_splits, loss_function, plot, all_outputs)


dataset = pd.read_csv('data/Data.csv', sep=';')
df = dataset[['Time', 'x013']]

test = TrainModel(nn_model=ModelLSTM(1, 5, 4), data=df, num_epochs=5, plot=True)

print(test.train_losses)
print(test.test_losses)

# train_losses = np.array(train_losses)
# test_losses = np.array(test_losses)
#
# signs = []
#
# for i in range(len(train_losses)):
#     mean = abs(train_losses - test_losses).mean()
#     std = abs(train_losses - test_losses).std()
#     if train_losses[i] - test_losses[i] > mean + 3*std or train_losses[i] - test_losses[i] < mean - 3*std:
#         print(f'{i}, significant')
#         signs.append(i)
#
# # вот у тя выдаются индексы теста теперь по каждому из них делай массив разностей, потом помечай аномалии, готово.
# # еще надо обернуть это все в функцию или класс по хорошему
# signs = np.array(signs)
# for i, (train_indexes, test_indexes) in enumerate(tscv.split(DataPrep(df, split_index=len(df)).X)):
#     for j in signs:
#         if i == j:
#             _, idxs = train_indexes, test_indexes
#
#
# print(f'Train MAPE: {train_losses}')
# print(f'Test MAPE: {test_losses}')
# plt.show()
