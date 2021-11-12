from numpy import array, transpose, zeros, std, mean
import pandas as pd
from matplotlib import pyplot as plt
from adtk.detector import ThresholdAD, LevelShiftAD, VolatilityShiftAD
from adtk.visualization import plot
from adtk.data import validate_series
# from tsfresh import extract_features

import test_function
import ts_analysis

#%% Time series definition
# [t, X] = test_function.solar_power_sso(1) 
# [t, X] = test_function.sin_function() 
# [t, X] = test_function.square_function() 
# [t, X] = test_function.cubic_function() 
# [t, X] = test_function.test_sine()
[t, X] = test_function.read("20211014.plt") 

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
decomposition = ts_analysis.ts_decomposition(df, plot=True, method='mean_value', noise_filter=True)

# Include decomposition in Dataframe 
df['trend'] = decomposition.trend 
df['seasonal'] = decomposition.seasonal
df['resid'] = decomposition.resid  

plt.show()

#%% Anomaly detection
anomaly_tag ={'th':'marker', 'ls':'span', 'vol':'span'} 
anomaly_color ={'th':'red', 'ls':'orange', 'vol':'green'}   

#Whole analysis
sigma_total = std(X)
mean_total = mean(X)
ts = validate_series(df['X(t)'])
 #Threshold: 3*sigma 
threshold_ad = ThresholdAD(high=mean_total + 3*sigma_total, low=mean_total - 3*sigma_total)
th_anomalies = threshold_ad.detect(ts)

 #Level shift 
level_shift_ad = LevelShiftAD(c=3.5, side='both', window=20)
ls_anomalies = level_shift_ad.fit_detect(ts)

 #Volatility
volatility_shift_ad = VolatilityShiftAD(c=12.0, side='both', window=30)
vol_anomalies = volatility_shift_ad.fit_detect(ts) 

ts_dict ={'th':th_anomalies, 'ls':ls_anomalies, 'vol':vol_anomalies} 

plot(ts, anomaly=ts_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=anomaly_color, anomaly_tag=anomaly_tag)

 
#Trend analysis
sigma_t = std(decomposition.trend)
mean_t = mean(decomposition.trend)
trend = validate_series(df['trend'])

 #Threshold: 3*sigma
threshold_ad = ThresholdAD(high=mean_t + 3*sigma_t, low=mean_t - 3*sigma_t)
th_anomalies = threshold_ad.detect(trend)

 #Level shift 
level_shift_ad = LevelShiftAD(c=4.0, side='both', window=20)
ls_anomalies = level_shift_ad.fit_detect(trend)

trend_dict ={'th':th_anomalies, 'ls':ls_anomalies} 
tag = {}
color = {}
for key in trend_dict.keys():
    color[key]  = anomaly_color[key]
    tag[key]  = anomaly_tag[key]

plot(trend, anomaly=trend_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=color, anomaly_tag=tag)

#Seasonal analysis
sigma_s = std(decomposition.seasonal)
mean_s = mean(decomposition.seasonal)
seasonal = validate_series(df['seasonal'])
 #Threshold: 3*sigma 
threshold_ad = ThresholdAD(high=mean_s + 3*sigma_s, low=mean_s - 3*sigma_s)
th_anomalies = threshold_ad.detect(seasonal)

 #Level shift 
level_shift_ad = LevelShiftAD(c=3.5, side='both', window=20)
ls_anomalies = level_shift_ad.fit_detect(seasonal)

seasonal_dict ={'th':th_anomalies, 'ls':ls_anomalies} 

 #Volatility 
volatility_shift_ad = VolatilityShiftAD(c=12.0, side='both', window=30)
if max(abs(seasonal)) > 0:
    vol_anomalies = volatility_shift_ad.fit_detect(seasonal)
    seasonal_dict['vol'] = vol_anomalies

tag = {}
color = {}
for key in seasonal_dict.keys():
    color[key]  = anomaly_color[key]
    tag[key]  = anomaly_tag[key]

plot(seasonal, anomaly=seasonal_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=color, anomaly_tag=tag)

#Residual analysis
sigma_r = std(decomposition.resid)
mean_r = mean(decomposition.resid)
resid = validate_series(df['resid'])
 #Threshold: 3*sigma 
threshold_ad = ThresholdAD(high=mean_r + 3*sigma_r, low=mean_r - 3*sigma_r)
th_anomalies = threshold_ad.detect(resid)

resid_dict ={'th':th_anomalies}
 #Volatility
volatility_shift_ad = VolatilityShiftAD(c=5.0, side='both', window=30)
if max(abs(resid)) > 0:
    vol_anomalies = volatility_shift_ad.fit_detect(resid)
    resid_dict['vol'] = vol_anomalies

tag = {}
color = {}
for key in resid_dict.keys():
    color[key]  = anomaly_color[key]
    tag[key]  = anomaly_tag[key]

plot(resid, anomaly=resid_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=color, anomaly_tag=tag)

plt.show()


# %%
