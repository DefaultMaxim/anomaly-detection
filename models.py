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

    def __init__(self, data, split_index: int, n_steps: int = 5):
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


dataset = pd.read_csv('data/Data.csv', sep=';')
df = dataset[['Time', 'x013']]
model = ModelLSTM(1, 8, 4)


def train_one_epoch(nn_model, train_loader):
    nn_model.train(True)

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


def validate_one_epoch(nn_model, test_loader):
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


learning_rate = 0.001
num_epochs = 5

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

tscv = TimeSeriesSplit(n_splits=5)

dd = []

train_losses = []
test_losses = []

for i, (train_index, test_index) in enumerate(tscv.split(DataPrep(df, split_index=len(df)).X)):

    dd.append(DataPrep(df[:test_index[-1]], train_index[-1]))

    for epoch in range(num_epochs):

        print(f'Epoch: {epoch + 1}')
        train_one_epoch(model, dd[i].train_loader)
        validate_one_epoch(model, dd[i].test_loader)

    with torch.no_grad():

        predicted = model(dd[i].X_train).numpy()

    train_predictions = predicted.flatten()

    dummies = np.zeros((dd[i].X_train.shape[0], dd[i].n_steps+1))
    dummies[:, 0] = train_predictions
    dummies = dd[i].scaler.inverse_transform(dummies)

    train_predictions = dc(dummies[:, 0])

    dummies = np.zeros((dd[i].X_train.shape[0], dd[i].n_steps+1))
    dummies[:, 0] = dd[i].y_train.flatten()
    dummies = dd[i].scaler.inverse_transform(dummies)

    new_y_train = dc(dummies[:, 0])

    test_predictions = model(dd[i].X_test).detach().numpy().flatten()

    dummies = np.zeros((dd[i].X_test.shape[0], dd[i].n_steps+1))
    dummies[:, 0] = test_predictions
    dummies = dd[i].scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((dd[i].X_test.shape[0], dd[i].n_steps+1))
    dummies[:, 0] = dd[i].y_test.flatten()
    dummies = dd[i].scaler.inverse_transform(dummies)

    new_y_test = dc(dummies[:, 0])

    # plt.plot(dd[i].y_train, label='Actual')
    # plt.plot(predicted, label='Predicted')
    # plt.xlabel('Day')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(new_y_test, label='Actual')
    # plt.plot(test_predictions, label='Predicted')
    # plt.xlabel('Day')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.show()

    train_losses.append(mape(new_y_train, train_predictions)*100)
    test_losses.append(mape(new_y_test, test_predictions)*100)

    print(f'Train MAPE: {mape(new_y_train, train_predictions)}, '
          f'Test MAPE: {mape(new_y_test, test_predictions)}')

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)

signs = []

for i in range(1, len(train_losses)):
    mean = abs(train_losses - test_losses).mean()
    std = abs(train_losses - test_losses).std()
    if abs(train_losses[i] - test_losses[i]) > mean + 3*std:
        print(f'{i}, significant')
        signs.append(i)

# вот у тя выдаются индексы теста теперь по каждому из них делай массив разностей, потом помечай аномалии, готово.
# еще надо обернуть это все в функцию или класс по хорошему
signs = np.array(signs)
for i, (train_indexes, test_indexes) in enumerate(tscv.split(DataPrep(df, split_index=len(df)).X)):
    for j in signs:
        if i == j:
            print(train_indexes, test_indexes)


print(f'Train MAPE: {train_losses}')
print(f'Test MAPE: {test_losses}')
plt.show()
