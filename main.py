import time
import os

from numpy import array, transpose, zeros, std, mean, load, linspace
import pandas as pd
from matplotlib import pyplot as plt

from adtk.data import validate_series

# from tsfresh import extract_features

import test_function
import ts_analysis
import ts_anomalies

#%% Time series definition

def main(filename, plot_figures, begin, end):
    # [t, X] = test_function.solar_power_sso(1) 
    # [t, X] = test_function.sin_function() 
    # [t, X] = test_function.square_function() 
    # [t, X] = test_function.cubic_function() 
    # [t, X] = test_function.test_sine()
    # [t, X] = test_function.read("20211014.plt") 
    # [t, X] = test_function.load_npy("P-11.npy") 
    # [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")
    [t, X] = test_function.read_UCR(filename)

    #Original time series plot
    if plot_figures:
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
    t0 = time.time()
    #Decomposition of the time series. Available methods: 'STL', 'seasonal_decompose', 'mean_value' and 'CNN
    decomposition, period = ts_analysis.ts_decomposition(df, plot=False, method='STL', noise_filter=True)
    tf = time.time()
    print('Time STL:', tf-t0)

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
    decomposition, period = ts_analysis.ts_decomposition(df_anomaly, plot=True, method='mean_value', noise_filter=True, period=period)
    tf = time.time()
    print('Time MVD:', tf-t0)

    # Include decomposition in Dataframe 
    df_anomaly['trend'] = decomposition.trend 
    df_anomaly['seasonal'] = decomposition.seasonal
    df_anomaly['resid'] = decomposition.resid  

    df_anomaly = validate_series(df_anomaly)


    #%% Anomaly detection
    #labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
    labels = ["ts", "trend", "seasonal", "resid"] 
    anomaly = ts_anomalies.Anomaly_detection(df_anomaly, labels, plot_anomalies=True)

    plt.figure()
    plt.plot(df['time'], df['X(t)'] ,'b')

    # plt.plot(df_anomaly['time'], anomaly.master_dict['minor'],'g.' )
    # plt.plot(df_anomaly['time'], anomaly.master_dict['significant'],'m.' )
    # plt.plot(df_anomaly['time'], anomaly.master_dict['major'],'r.' )
    value = 0
    color ={'minor':'g', 'significant':'m', 'major':'r'}  
    for key in anomaly.master_dict.keys():
        aux_t =[]
        aux_anomaly =[]
        ct = 0

        for i in range(0, len(anomaly.master_dict[key])):
            if anomaly.master_dict[key][i] == 1:
                # if ct == 1 and key == 'significant' and plot_figures:
                #     plt.axvspan(df_anomaly['time'][i-10] , df_anomaly['time'][i+10], facecolor=color[key], alpha=0.5)
                aux_t.append(df_anomaly['time'][i])  
                aux_anomaly.append(df_anomaly['X(t)'][i])
                ct = 1
            else:
                # if ct == 1 and key == 'significant' and plot_figures:
                    # plt.axvspan(df_anomaly['time'][i-5] , df_anomaly['time'][i+5], facecolor=color[key], alpha=0.5)
                ct = 0
        
        if key == 'major':
            if plot_figures:
                plt.plot(aux_t, aux_anomaly, 'ro')
            for timestamp in aux_t:
                if timestamp > begin and timestamp < end:
                    value = 3
        elif key == 'significant' and value < 2:
            for timestamp in aux_t:
                if timestamp > begin and timestamp < end:
                    value = 2
        elif key == 'minor':
            if plot_figures:
                plt.plot(aux_t, aux_anomaly, '.g')
            if value < 1:
                for timestamp in aux_t:
                    if timestamp > begin and timestamp < end:
                        value = 1

    if plot_figures:
        plt.legend(['Time series Anomalies', 'major', 'minor', 'significant'])
        plt.show()

    return value


if __name__ == "__main__":
    # path = os.getcwd() + '/UCR_Anomaly_FullData'
    # dir_list = os.listdir(path)
    # f = open('results.txt', 'w')
    # f.write('--- Major = 3, significant = 2, minor = 1, not found = 0--- \n')
    # f.write('\n')
    # val_ct = 0
    # ct = 0
    # major_ct = 0
    # significant_ct = 0
    # minor_ct = 0
    # for dir_file in dir_list:
    #     ct += 1
    #     split = dir_file.split('_')
    #     begin = int(split[-2])
    #     end = int(split[-1].split('.')[0])  
    #     print('Anomaly' + str(ct)+':',begin,'-',end)

    #     value = main(path + '/' + dir_file, False, begin, end)
    #     val_ct += value
    #     if value == 3:
    #         major_ct += 1
    #     elif value == 2:
    #         significant_ct += 1
    #     elif value == 1:
    #         minor_ct += 1
    #     print('Value:', value, '\n')
    #     f.write(str(ct) + ') ' + dir_file + ', ' + str(value) + '\n')

    # print('Result:', val_ct, '/', 3*ct)
    # f.write('\n')
    # f.write('Result: ' + str(val_ct) + '/' + str(3*ct))
    # f.write('\n')
    # f.write('Major: ' + str(major_ct) + ', Significant: ' + str(significant_ct) + ', Minor: ' + str(minor_ct))
    # f.close()

    value = main("UCR_Anomaly_FullData/146_UCR_Anomaly_Lab2Cmac011215EPG2_5000_27862_27932.txt", True, 27862, 27932)