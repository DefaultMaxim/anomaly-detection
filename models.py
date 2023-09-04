import numpy as np
import pandas as pd
from collections import namedtuple


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

            if data[key] > high[key] or data[key] < low[key]:
                anomalies[key] = True

        ntup = namedtuple('Bounds', ['high', 'low'])

        bounds = ntup(high, low)

        return pd.Series(anomalies), bounds

    else:

        mean = data.mean()
        std = data.std()

        high = mean + threshold*std
        low = mean - threshold*std

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

            if data[key] > high[key] or data[key] < low[key]:

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
