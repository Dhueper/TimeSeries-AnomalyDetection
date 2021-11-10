from numpy import array, transpose, zeros, std, mean
import pandas as pd
from matplotlib import pyplot as plt
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series
# from tsfresh import extract_features

import test_function
import ts_analysis

#%% Time series definition
[t, X] = test_function.solar_power_sso(1) 
# [t, X] = test_function.sin_function() 
# [t, X] = test_function.square_function() 
# [t, X] = test_function.cubic_function() 
# [t, X] = test_function.test_sine()
# [t, X] = test_function.read("20211014.plt") 

#Original time series plot
plt.figure()
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Original time series')

datetime = pd.to_datetime(list(t), unit="s")

id_column = zeros(len(t))
time_series = transpose(array([t,X,id_column]))

df = pd.DataFrame(time_series, columns=["time", "X(t)", "id"]) 

df["datetime"] = datetime 

# Feature extraction with tsfresh 
# features = extract_features(df, column_id='id', column_sort='time')
# print(features)

df.set_index("datetime", inplace=True) 

#Decomposition of the time series. Available methods: 'STL', 'seasonal_decompose', 'mean_value' and 'CNN
decomposition = ts_analysis.ts_decomposition(df, plot=False, method='mean_value', noise_filter=False)

# Include decomposition in Dataframe 
df['trend'] = decomposition.trend 
df['seasonal'] = decomposition.seasonal
df['residuals'] = decomposition.resid  

plt.show()

#%% Anomaly detection
 
#Trend analysis
sigma_t = std(decomposition.trend)
mean_t = mean(decomposition.trend)

threshold_ad = ThresholdAD(high=mean_t + 3*sigma_t, low=mean_t - 3*sigma_t)
trend = validate_series(df['trend'])
th_anomalies = threshold_ad.detect(trend)

plot(trend, anomaly=th_anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

plt.show()