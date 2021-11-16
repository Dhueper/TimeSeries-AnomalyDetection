import time

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
[t, X] = test_function.load_npy("P-11.npy") 
# [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")

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

#%% Anomaly detection
#labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
labels = ["ts"] 
anomaly = ts_anomalies.Anomaly_detection(df, labels)
anomaly_list = array([False for _ in range(0,len(X))])

X_anomaly = []  
t_anomaly = [] 

for key in anomaly.ts_dict.keys():
    anomaly_list = anomaly_list | anomaly.ts_dict[key]

aux = len(X)
for i in range(int(len(X)/10),int(9*len(X)/10)):
    if anomaly_list[i] and aux >= int(len(X)/100):
        aux = 1
        for j in range(i-int(len(X)/100), i+1):
            X_anomaly.append(X[j])
            t_anomaly.append(t[j])
    elif anomaly_list[i] and aux < int(len(X)/100):
        aux = 1
        X_anomaly.append(X[i])
        t_anomaly.append(t[i])
    elif aux < int(len(X)/100):
        aux += 1
        X_anomaly.append(X[i])
        t_anomaly.append(t[i])

plt.figure()
plt.plot(t_anomaly,X_anomaly) 

plt.show()

#%% Deomposition 
time_series = transpose(array([t_anomaly,X_anomaly]))

df_anomaly = pd.DataFrame(time_series, columns=["time", "X(t)"]) 

datetime = pd.to_datetime(list(t_anomaly), unit="s")
df_anomaly["datetime"] = datetime 

df_anomaly.set_index("datetime", inplace=True) 

t0 = time.time()
#Decomposition of the time series. Available methods: 'STL', 'seasonal_decompose', 'mean_value' and 'CNN
decomposition = ts_analysis.ts_decomposition(df_anomaly, plot=True, method='mean_value', noise_filter=True)
tf = time.time()
print('Time:', tf-t0)

# Include decomposition in Dataframe 
df_anomaly['trend'] = decomposition.trend 
df_anomaly['seasonal'] = decomposition.seasonal
df_anomaly['resid'] = decomposition.resid  

plt.show()

#%% Anomaly detection
#labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
labels = ["ts", "trend", "seasonal", "resid"] 
anomaly = ts_anomalies.Anomaly_detection(df_anomaly, labels)

plt.figure()
plt.plot(t_anomaly, X_anomaly,'b')
plt.plot(t_anomaly, anomaly.master_dict['major'],'r.' )

plt.show()


