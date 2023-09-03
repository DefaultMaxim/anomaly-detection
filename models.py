import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
from utils import anomalies_plot


def const_std_model(data, threshold: int = 3):

    """

    detect anomalies based on threshold*sigma rule
    :param data: data
    :param threshold: std threshold (usual from 2 to 5)
    :return: anomalies : dataframe,
    bounds: namedtuple(high, low)

    """

    mean = data.mean()
    std = data.std()

    high = mean + threshold*std
    low = mean - threshold*std

    boarders = namedtuple('Bounds', ['high', 'low'])
    bounds = boarders(high, low)

    anomalies = pd.concat([data > high, data < low], axis=1).any(axis=1)

    return pd.Series(anomalies), bounds


def const_iqr_model(data, threshold=3):

    """

    inter quartile range model
    :param data:data
    :param threshold:model threshold
    :return:anomalies, bounds

    """

    iqr = data.quantile(0.75) - data.quantile(0.25)

    high = data.quantile(0.75) + (iqr * threshold)
    low = data.quantile(0.25) - (iqr * threshold)

    ntup = namedtuple('Bounds', ['high', 'low'])

    bounds = ntup(high, low)

    anomalies = pd.concat([data > high, data < low], axis=1).any(axis=1)

    return pd.Series(anomalies), bounds


def roll_std_model(data, threshold=3):

    """
    Rolling std model, std changes over time
    :param data: data
    :param threshold: model threshold
    :return: True/False series of anomalies

    """

    anomalies = np.array([False]*len(data))

    high = np.array([0] * len(data), dtype='float')
    low = np.array([0] * len(data), dtype='float')

    for key, _ in enumerate(data):

        mean = data[:key].mean()
        std = data[:key].std()

        high[key] = mean + threshold * std
        low[key] = mean - threshold * std

        if data[key] > high[key] or data[key] < low[key]:

            anomalies[key] = True

    ntup = namedtuple('Bounds', ['high', 'low'])

    bounds = ntup(high, low)

    return pd.Series(anomalies), bounds


def roll_iqr_model(data, threshold=3):

    """

    Rolling iqr model where iqr changes over time
    :param data: data
    :param threshold: model threshold
    :return: True/False series of anomalies

    """

    anomalies = np.array([False] * len(data))

    high = np.array([0]*len(data), dtype='float')
    low = np.array([0]*len(data), dtype='float')

    for key, _ in enumerate(data):

        iqr = data[:key].quantile(0.75) - data[:key].quantile(0.25)

        high[key] = data[:key].quantile(0.75) + (iqr * threshold)

        low[key] = data[:key].quantile(0.25) - (iqr * threshold)

        if data[key] > high[key] or data[key] < low[key]:

            anomalies[key] = True

    ntup = namedtuple('Bounds', ['high', 'low'])

    bounds = ntup(high, low)

    return pd.Series(anomalies), bounds


df = pd.read_csv('data/Data.csv', sep=';')

series = df.x013

an, bds = roll_iqr_model(series, threshold=2)

anomalies_plot(series, an, bds)
