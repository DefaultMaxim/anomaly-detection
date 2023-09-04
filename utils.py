import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple


def get_column_outliers(data, model, threshold=3):

    """

    Gets True/False array of anomalies where True index is anomaly index
    :param data: data
    :param model: anomaly detection model
    :param threshold: anomaly detection threshold
    :return: anomaly series

    """

    outliers = pd.Series(data=[False] * len(data), name='is_outlier')

    anomalies, bounds = model(data, threshold=threshold)

    outliers[anomalies[anomalies].index] = True

    return outliers


def anomalies_report(outliers):

    """

    gets anomalies statistics
    :param outliers: anomalies series
    :return: anomaly statistics

    """

    print(f"Total number of outliers: {sum(outliers)}\n"
          f"Percentage of outliers:   {np.round(100 * sum(outliers) / len(outliers), 4)}%")


def anomalies_plot(data, anomalies, bounds: namedtuple):

    """

    plot anomalies points on time series

    :param data: data time series
    :param anomalies: True/False array where True index is anomaly index
    :param bounds: extr frontiers
    :return: plot

    """

    figure, ax = plt.subplots(figsize=(14, 8))

    plt.plot(data)

    if type(bounds.high) == np.ndarray:

        plt.plot(bounds.high, color='red')

        plt.plot(bounds.low, color='red')

    else:

        plt.axhline(bounds.high, color='red')

        plt.axhline(bounds.low, color='red')

    for i in range(len(anomalies)):

        if anomalies[i]:

            ax.scatter(i, data[i], color="red", marker='x', label='1880-1999')

    plt.show()
