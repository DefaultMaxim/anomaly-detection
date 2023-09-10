# Anomaly-detection
---
Project for anomaly detection in time series.
---
Technologies stack:
---
1. NumPy
2. Pandas
3. PyTorch
4. Matplotlib (for visualization)
---
Example std-anomaly detection model:
---
This method better works when $Data \sim \mathcal{N}(\mu,\,\sigma^{2})$, if your data is not normal: better use IQR model or LSTM model
1. Firstly, gets data, for example data/Data.csv contains parameters data of hydroelectric power station.
2. Then, calculates mean, std error, if parameter roll = True, then we compute rolling mean, rolling std across all time series.
3. After this, if $data_i > mean + threshold \cdot std$ or $data_i < mean - threshold \cdot std$ then $data_i$ is anomaly.
   
