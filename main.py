import time

from numpy import array, transpose, zeros, std, mean, load, linspace
import pandas as pd
from matplotlib import pyplot as plt

from adtk.data import validate_series

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

#%% Decomposition STL
#Decomposition of the time series. Available methods: 'STL', 'seasonal_decompose', 'mean_value' and 'CNN
decomposition = ts_analysis.ts_decomposition(df, plot=False, method='STL', noise_filter=True)

# Include decomposition in Dataframe 
df['trend'] = decomposition.trend 
df['seasonal'] = decomposition.seasonal
df['resid'] = decomposition.resid  

#%% Anomaly detection
#labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
labels = ["ts", "trend", "seasonal", "resid"] 
anomaly = ts_anomalies.Anomaly_detection(df, labels, plot_anomalies=False)
anomaly_list = array([False for _ in range(0,len(X))])

X_anomaly = []  
t_anomaly = [] 

for key in anomaly.master_dict.keys():
    anomaly_list = anomaly_list | array(anomaly.master_dict[key], dtype=bool)

aux = len(X)
ct = 0
for i in range(int(len(X)/10),int(9*len(X)/10)):
    if anomaly_list[i] and aux >= int(len(X)/50):
        aux = 1
        if ct == 0:
            ct = 1
            for j in range(i-int(len(X)/50), i+1):
                X_anomaly.append(X[j])
                t_anomaly.append(t[j])
        else:
            for j in range(i-int(len(X)/50), i+1):
                X_anomaly.append(X[j])
                t_anomaly.append(t[j])
    elif anomaly_list[i] and aux < int(len(X)/50):
        aux = 1
        X_anomaly.append(X[i])
        t_anomaly.append(t[i])
    elif aux < int(len(X)/50):
        aux += 1
        X_anomaly.append(X[i])
        t_anomaly.append(t[i])

print('Original length =',len(X), ', New length =', len(X_anomaly))

# plt.figure()
# plt.plot(t_anomaly,X_anomaly) 
# plt.show()

#%% Deomposition 
time_series = transpose(array([t_anomaly,X_anomaly]))

df_anomaly = pd.DataFrame(time_series, columns=["time", "X(t)"]) 

datetime = pd.to_datetime(list(t_anomaly), unit="s")
df_anomaly["datetime"] = datetime 

df_anomaly.set_index("datetime", inplace=True) 

t0 = time.time()
#Decomposition of the time series. Available methods: 'STL', 'seasonal_decompose', 'mean_value' and 'CNN
decomposition = ts_analysis.ts_decomposition(df_anomaly, plot=False, method='mean_value', noise_filter=True)
tf = time.time()
print('Time:', tf-t0)

# Include decomposition in Dataframe 
df_anomaly['trend'] = decomposition.trend 
df_anomaly['seasonal'] = decomposition.seasonal
df_anomaly['resid'] = decomposition.resid  

df_anomaly = validate_series(df_anomaly)


plt.show()

#%% Anomaly detection
#labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
labels = ["ts", "trend", "seasonal", "resid"] 
anomaly = ts_anomalies.Anomaly_detection(df_anomaly, labels, plot_anomalies=False)

plt.figure()
plt.plot(df['time'], df['X(t)'] ,'b')

# plt.plot(df_anomaly['time'], anomaly.master_dict['minor'],'g.' )
# plt.plot(df_anomaly['time'], anomaly.master_dict['significant'],'m.' )
# plt.plot(df_anomaly['time'], anomaly.master_dict['major'],'r.' )

color ={'minor':'g', 'significant':'m', 'major':'r'}  
for key in anomaly.master_dict.keys():
    aux_t =[]
    aux_anomaly =[]
    ct = 0

    for i in range(0, len(anomaly.master_dict[key])):
        if anomaly.master_dict[key][i] == 1:
            if ct == 1 and key == 'significant':
                plt.axvspan(df_anomaly['time'][i-10] , df_anomaly['time'][i+10], facecolor=color[key], alpha=0.5)
            aux_t.append(df_anomaly['time'][i])  
            aux_anomaly.append(df_anomaly['X(t)'][i])
            ct = 1
        else:
            if ct == 1 and key == 'significant':
                plt.axvspan(df_anomaly['time'][i-5] , df_anomaly['time'][i+5], facecolor=color[key], alpha=0.5)
            ct = 0
    
    if key == 'major':
        plt.plot(aux_t, aux_anomaly, 'ro')
    elif key == 'minor':
        plt.plot(aux_t, aux_anomaly, '.g')




plt.legend(['Time series Anomalies', 'major', 'minor', 'significant'])

plt.show()


