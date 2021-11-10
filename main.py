from numpy import array, transpose, zeros, std, mean
import pandas as pd
from matplotlib import pyplot as plt
from adtk.detector import ThresholdAD, LevelShiftAD
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
decomposition = ts_analysis.ts_decomposition(df, plot=True, method='mean_value', noise_filter=False)

# Include decomposition in Dataframe 
df['trend'] = decomposition.trend 
df['seasonal'] = decomposition.seasonal
df['resid'] = decomposition.resid  

plt.show()

#%% Anomaly detection

#Whole analysis
sigma_total = std(X)
mean_total = mean(X)
ts = validate_series(df['X(t)'])

threshold_ad = ThresholdAD(high=mean_total + 3*sigma_total, low=mean_total - 3*sigma_total)
th_anomalies = threshold_ad.detect(ts)

plot(ts, anomaly=th_anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

 
#Trend analysis
sigma_t = std(decomposition.trend)
mean_t = mean(decomposition.trend)
trend = validate_series(df['trend'])

 #Threshold: 3*sigma
threshold_ad = ThresholdAD(high=mean_t + 3*sigma_t, low=mean_t - 3*sigma_t)
th_anomalies = threshold_ad.detect(trend)

 #Level shift 
level_shift_ad = LevelShiftAD(c=4.0, side='both', window=5)
ls_anomalies = level_shift_ad.fit_detect(trend)

# trend_series = transpose(array([th_anomalies, ls_anomalies]))

# trend_df = pd.DataFrame(trend_series, columns=["th", "ls"])
trend_dict ={'th':th_anomalies, 'ls':ls_anomalies} 

plot(trend, anomaly=trend_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

#Seasonal analysis
sigma_s = std(decomposition.seasonal)
mean_s = mean(decomposition.seasonal)
seasonal = validate_series(df['seasonal'])
threshold_ad = ThresholdAD(high=mean_s + 3*sigma_s, low=mean_s - 3*sigma_s)
th_anomalies = threshold_ad.detect(seasonal)

plot(seasonal, anomaly=th_anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

#Residual analysis
sigma_r = std(decomposition.resid)
mean_r = mean(decomposition.resid)
resid = validate_series(df['resid'])
threshold_ad = ThresholdAD(high=mean_r + 3*sigma_r, low=mean_r - 3*sigma_r)
th_anomalies = threshold_ad.detect(resid)

plot(resid, anomaly=th_anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")


plt.show()


# %%
