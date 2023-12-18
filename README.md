# Anomaly-detection
---
Project for anomaly detection in time series.
---
Technologies stack:
---
1. NumPy (ver. 1.24.2)
2. Pandas (ver. 1.5.3)
3. PyTorch (ver. 2.0.1+cpu)
4. Matplotlib (for visualization)
---
Example std-anomaly detection model:
---
This method better works when $Data \sim \mathcal{N}(\mu,\sigma^{2})$, if your data is not normal: better use IQR model or LSTM model
1. Firstly, gets data, for example data/Data.csv contains parameters data of hydroelectric power station.
2. Then, calculates mean, std error, if parameter roll = True, then we compute rolling mean, rolling std across all time series.
3. After this, if $data_i > mean + threshold \cdot std$ or $data_i < mean - threshold \cdot std$ then $data_i$ is anomaly.
   
Example of work std model:

__Red x means anomalies, red lines means critical threshold bounds.__

![std](https://github.com/DefaultMaxim/anomaly-detection/blob/master/examples/std_anomaly.png?raw=true)
---
Example LSTM-anomaly detection model:
---
1. After we get data, algorithm makes n_splits from it.
2. On each split from n_splits we make train/test data, and train/validate our model.
3. On eeach split we compute train mape and test mape, where ratio $\frac{test_{mape}}{train_{mape}}$ is outlier (we can check it via iqr or std model) there is anomaly.

__Example of work LSTM model:__

__Red x means anomalies.__

![lstm](https://github.com/DefaultMaxim/anomaly-detection/blob/master/examples/lstm_anomaly.png?raw=true)

__Example SBERP anomaly detection via LSTM - model:__

__Red x means anomalies.__

![SBERP](https://github.com/DefaultMaxim/anomaly-detection/assets/112869928/b800d2bc-52e9-49fc-9ec2-97ba0823ad1b)



