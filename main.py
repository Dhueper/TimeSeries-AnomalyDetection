from numpy import array, transpose, zeros, std, mean, load, linspace
import pandas as pd
from matplotlib import pyplot as plt

# from tsfresh import extract_features

import test_function
import ts_analysis
import ts_anomalies

#%% Time series definition
# [t, X] = test_function.solar_power_sso(1) 
# [t, X] = test_function.sin_function() 
# [t, X] = test_function.square_function() 
# [t, X] = test_function.cubic_function() 
# [t, X] = test_function.test_sine()
# [t, X] = test_function.read("20211014.plt") 
# [t, X] = test_function.load_npy("P-11.npy") 
[t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")

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
#labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
labels = ["ts", "trend", "seasonal", "resid"] 
anomaly = ts_anomalies.Anomaly_detection(df, labels)

plt.show()

# %%
