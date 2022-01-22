import time
import os
import sys
import json

from numpy import array, transpose, zeros, std, mean, load, linspace, log, exp, ones, angle, convolve, sqrt, pi, sum, argmin, asfortranarray
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
from matplotlib import pyplot as plt

from adtk.data import validate_series

# from tsfresh import extract_features

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/fortran_interface')
    import fortran_ts
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/fortran_interface')
    import fortran_ts

try:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))+'/sources')
except:
    sys.path.insert(1, '/'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))+'/sources')

import test_function
import ts_analysis
import ts_anomalies
import plots

#%% Time series definition

def main(filename, plot_figures, begin, end, data):
    """Main program to execute the analysis, decomposition and anomaly detection
    in time series. The code is optimized to minimize the time required.

    No need to estimate parameters, everything is calculated automatically.

    Configured for NASA's and UCR's database. 
    Feel free to add your own anomaly detection database after line 50!

    Input:
        filename (string), file name with the data;
        plot_figures (boolean), decides whether to plot the figures or not 
        (False recommended for large database evaluation);
        begin (list), list which contains the initial timestamp for each anomaly;
        end (list), list which contains the final timestamp for each anomaly;
        data (string), selected database (NASA or UCR).
    """

    if data == "UCR":
        [t, X] = test_function.read_UCR(filename)
    elif data == "NASA":
        [t, X] = test_function.load_npy(filename) 

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

    #Spectral residual
    df['sr'] = spectral_residual(X, 1)

    # plt.figure()
    # plt.plot(t, df['sr'])   
    # plt.show()

    #%% Anomaly detection
    #labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid" 
    labels = ["ts", "trend", "seasonal", "resid", "sr"] 
    anomaly = ts_anomalies.Anomaly_detection(df, labels, plot_anomalies=False)

    anomaly_list = array([False for _ in range(0,len(X))])

    X_anomaly = []  
    t_anomaly = [] 

    for key in anomaly.master_dict.keys():
        anomaly_list = anomaly_list | array(anomaly.master_dict[key], dtype=bool)

    aux = len(X)
    ct = 0
    for i in range(int(len(X)/20),int(19*len(X)/20)):
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

    if len(X_anomaly) < 10:
        X_anomaly = [] 
        t_anomaly = [] 
        for i in range(int(len(X)/20),int(19*len(X)/20)):
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
    decomposition, period = ts_analysis.ts_decomposition(df_anomaly, plot=False, method='mean_value', noise_filter=True, period=int(period))
    tf = time.time()
    print('Time MVD:', tf-t0)

    # Include decomposition in Dataframe 
    df_anomaly['trend'] = decomposition.trend 
    df_anomaly['seasonal'] = decomposition.seasonal
    df_anomaly['resid'] = decomposition.resid  

    #Spectral residual
    df_anomaly['sr'] = spectral_residual(X_anomaly, 2) 

    df_anomaly = validate_series(df_anomaly)


    #%% Anomaly detection
    #labels to detect anomalies: "ts" (whole time series), "trend", "seasonal", "resid", 
    labels = ["ts", "trend", "seasonal", "resid", "sr"] 
    anomaly = ts_anomalies.Anomaly_detection(df_anomaly, labels, plot_anomalies=False)

    if plot_figures:
        plt.figure()
        plt.plot(df['time'], df['X(t)'] ,'b')

    # plt.plot(df_anomaly['time'], anomaly.master_dict['minor'],'g.' )
    # plt.plot(df_anomaly['time'], anomaly.master_dict['significant'],'m.' )
    # plt.plot(df_anomaly['time'], anomaly.master_dict['major'],'r.' )
    value = zeros(len(begin)) 
    val = 0
    color ={'minor':'g', 'significant':'m', 'major':'r'}  
    legend = ['Time series Anomalies']
    for key in anomaly.master_dict.keys():
        legend.append(key)
        aux_t =[]
        aux_anomaly =[]
        ct = 0

        # for i in range(0, len(anomaly.master_dict[key])):
        for i in range(0, len(df_anomaly['time'])):
            if anomaly.master_dict[key][i] == 1:
                # if ct == 1 and key == 'significant' and plot_figures:
                #     plt.axvspan(df_anomaly['time'][i-10] , df_anomaly['time'][i+10], facecolor=color[key], alpha=0.5)
                aux_t.append(df_anomaly['time'][i])  
                aux_anomaly.append(df_anomaly['X(t)'][i])
                ct = 1
            else:
                # if ct == 1 and key == 'significant' and plot_figures:
                #     plt.axvspan(df_anomaly['time'][i-5] , df_anomaly['time'][i+5], facecolor=color[key], alpha=0.5)
                ct = 0
        
        for i in range(0,len(begin)):
            b = begin[i] 
            e = end[i] 
            if key == 'major':
                for timestamp in aux_t:
                    if timestamp > b and timestamp < e:
                        value[i]  = 3
                        break

            elif key == 'significant' and value[i] < 2:
                for timestamp in aux_t:
                    if timestamp > b and timestamp < e:
                        value[i] = 2
                        break

            elif key == 'minor':
                if value[i] < 1:
                    for timestamp in aux_t:
                        if timestamp > b and timestamp < e:
                            value[i] = 1
                            break

        if key == 'major':
            if plot_figures:
                plt.plot(aux_t, aux_anomaly, 'ro')
        elif key == 'significant':
            if plot_figures:
                plt.plot(aux_t, aux_anomaly, 'mo')
        elif key == 'minor':
            if plot_figures:
                plt.plot(aux_t, aux_anomaly, '.g') 


    #Detect which method is more successfull  
    best_detection = [] 
    for i in range(0,len(df_anomaly['time'])):
        for limit in range(0,len(begin)):
            b = begin[limit] 
            e = end[limit]
            timestamp = df_anomaly['time'][i]
            if timestamp > b and timestamp < e:
                for j in anomaly.ts_dict.keys():
                    if (anomaly.ts_dict[j][i] == True) and ('ts' not in best_detection):
                        best_detection.append('ts') 
                for j in anomaly.sr_dict.keys():
                    if (anomaly.sr_dict[j][i] == True) and ('sr' not in best_detection):
                        best_detection.append('sr') 
                for j in anomaly.trend_dict.keys():
                    if (anomaly.trend_dict[j][i] == True) and ('trend' not in best_detection):
                        best_detection.append('trend') 
                for j in anomaly.seasonal_dict.keys():
                    if (anomaly.seasonal_dict[j][i] == True) and ('seasonal' not in best_detection):
                        best_detection.append('seasonal') 
                for j in anomaly.resid_dict.keys():
                    if (anomaly.resid_dict[j][i] == True) and ('resid' not in best_detection):
                        best_detection.append('resid') 

    if plot_figures:
        plt.legend(legend)
        plt.xlabel('t')
        plt.ylabel('X(t)', rotation=0)
        plt.show()

    val = sum(value) / len(begin)

    return val, best_detection

def spectral_residual(X, c):
    """Computes the spectral residual transformation of the time series.

    Input:
        X (numpy.array), time series;
        c (integer), exponential factor.

    Returns: S (numpy.array), transformed time series.

    """
    X_FFT = fft(X)
    A = abs(X_FFT) + 1e-8
    P = angle(X_FFT)
    L = log(A)
    q = 3
    hq = ones(q)/q
    R = L - convolve(L,hq, 'same')
    S = abs(ifft(exp(R + 1j*P))**c)

    return S

def user_examples(N):
    """Pre-defined examples to introduce a new user for time series analysis:
    1) Noise reduction.
    2) Period estimation.
    3) Time series decomposition: STL.
    4) Time series decomposition: MVD.
    5) Spectral residual transformation.
    6) Anomaly detection with ADTK.
    7) Time series decomposition and anomaly detection with ADTK.
    8) NASA's anomaly database evaluation (highly time consuming).
    9) UCR's anomaly database evaluation (highly time consuming).
    10) Comparison between computational time for python and fortran filter.
    

    Intent(in): N(integer), example selected;

    Returns: None
    """
    def plot(t,X):
        plt.figure()
        plt.plot(t,X)
        plt.xlabel('t')
        plt.ylabel('X(t)')
        plt.title('Original time series')

    def example1():
        """Time series noise reduction.

        Intent(in): None

        Returns: None
        """

        print('Example 1: Noise reduction filter.')

        [t, X] = test_function.solar_power_sso(1) 
        plot(t,X)

        #Noise Mean Value Filter
        Y = zeros(len(X))
        Y[:] = X[:]  
        for _ in range(0,10):
            Y = fortran_ts.time_series.mvf(asfortranarray(Y), 2)
            Y[0] = 2*Y[1] - Y[2] 
            Y[len(Y)-1] = 2*Y[len(Y)-2] - Y[len(Y)-3] 

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t,Y, 'g')
        plt.title('Noiseless time series')
        plt.xlabel('t')
        plt.ylabel('Y(t)', rotation=0)
        plt.subplot(2,1,2)
        plt.plot(t, X-Y, 'r')
        plt.title('Noise')
        plt.xlabel('t')
        plt.ylabel('N(t)', rotation=0)
        
        plt.show()

    def example2():
        """Time series period estimation.

        Intent(in): None

        Returns: None
        """

        print('Example 2: Time series period estimation.')

        [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")
        plot(t,X)

        #Period estimation 
        Ns = len(t)
        delta_t = t[1] - t[0]  

        dxs = zeros(Ns)
        dxs[0] = (X[1] - X[0])/delta_t
        dxs[Ns-1] = (X[Ns-1] - X[Ns-2])/delta_t
        for i in range(1,Ns-1):
            dxs[i] = (X[i+1] - X[i-1])/(2*delta_t) 

        N_vec = array([i for i in range(8,Ns//10)])
        E = [] 
        for N in N_vec: 

            #First N samples 
            x = X[0:N] 
            dx = dxs[0:N] 

            #Spectral domain
            xf = fft(x, N)
            f = fftfreq(N,delta_t)  

            dx_spec = ifft(2*pi*f*1j*xf, N).real 

            #Error 
            E.append(sqrt(sum((dx - dx_spec)**2.) / N))

        plt.figure()
        plt.plot(N_vec, E)
        plt.title('Error as a function of the number of samples N')
        plt.xlabel('N samples')
        plt.ylabel('Error')

        plt.show()

        print('The period (measured in number of points) is:', N_vec[argmin(E)])

    def example3():
        """Time series decomposition with STL method.

        Intent(in): None

        Returns: None
        """

        print('Example 3: Time series decomposition with STL.')

        [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/146_UCR_Anomaly_Lab2Cmac011215EPG2_5000_27862_27932.txt")
        t = t[0:2000]
        X = X[0:2000]  
        plot(t,X)

        datetime = pd.to_datetime(list(t), unit="s")

        id_column = zeros(len(t))
        time_series = transpose(array([t,X,id_column]))

        df = pd.DataFrame(time_series, columns=["time", "X(t)", "id"]) 

        df["datetime"] = datetime 

        df.set_index("datetime", inplace=True) 

        #  STL decomposition
        decomposition, period = ts_analysis.ts_decomposition(df, plot=True, method='STL', noise_filter=True) 

        plt.show()

    def example4():
        """Time series decomposition with MVD method.

        Intent(in): None

        Returns: None
        """

        print('Example 4: Time series decomposition with MVD.')

        [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/146_UCR_Anomaly_Lab2Cmac011215EPG2_5000_27862_27932.txt")
        t = t[0:2000]
        X = X[0:2000]  
        plot(t,X)

        datetime = pd.to_datetime(list(t), unit="s")

        id_column = zeros(len(t))
        time_series = transpose(array([t,X,id_column]))

        df = pd.DataFrame(time_series, columns=["time", "X(t)", "id"]) 

        df["datetime"] = datetime 

        df.set_index("datetime", inplace=True) 

        #  STL decomposition
        decomposition, period = ts_analysis.ts_decomposition(df, plot=True, method='mean_value', noise_filter=True) 

        plt.show()

    def example5():
        """Spectral residual transformation.

        Intent(in): None

        Returns: None
        """

        print('Example 5: Spectral residual transformation.')

        # [t, X] = test_function.load_npy("P-11.npy")
        [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/159_UCR_Anomaly_TkeepSecondMARS_3500_9330_9340.txt")
 
        plot(t,X)

        S = spectral_residual(X, 2)

        plt.figure()
        plt.plot(t, S, 'g')
        plt.xlabel("t")
        plt.ylabel("S", rotation=0)
        plt.title('Spectral residual transformation')

        plt.show()

    def example6():
        """Anomaly detection with ADTK.

        Intent(in): None

        Returns: None
        """

        print('Example 6: Anomaly detection with ADTK.')

        [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260.txt")
        t = t[16000:19000]
        X = X[16000:19000]  
        plot(t,X)

        datetime = pd.to_datetime(list(t), unit="s")

        id_column = zeros(len(t))
        time_series = transpose(array([t,X,id_column]))

        df = pd.DataFrame(time_series, columns=["time", "X(t)", "id"]) 

        df["datetime"] = datetime 

        df.set_index("datetime", inplace=True) 

        labels = ["ts"] 
        anomaly = ts_anomalies.Anomaly_detection(df, labels, plot_anomalies=True)


        plt.show()

    def example7():
        """Time series decomposition and anomaly detection with ADTK.

        Intent(in): None

        Returns: None
        """

        print('Example 7: Time series decomposition and anomaly detection with ADTK.')

        value, best_detection = main("UCR_Anomaly_FullData/098_UCR_Anomaly_NOISEInternalBleeding16_1200_4187_4199.txt", True, [4150], [4199], "UCR")

        print('Value:',value,'/3,  Best detection:', best_detection)

    def example8():
        """NASA's anomaly database evaluation.

        Intent(in): None

        Returns: None
        """

        print("Example 8: NASA's anomaly database evaluation (highly time consuming).")

        #--- NASA anomalies ---
         
        path = os.getcwd() + '/test_NASA'
        dir_list = os.listdir(path)

        f_anomalies = open("NASA_anomalies.txt", 'r')
        dict_an = {} 
        L = len(f_anomalies.readlines())
        f_anomalies.seek(0)
        for i in range(0,L):
            line = f_anomalies.readline().split(";")
            dict_an[line[0]] = json.loads(line[1].split("\n")[0])
        f_anomalies.close()
        keys = dict_an.keys()

        f = open('results.txt', 'w')
        f.write('NASA telemetry data: \n')
        f.write('--- Major = 3, significant = 2, minor = 1, not found = 0--- \n')
        f.write('\n')
        val_ct = 0
        ct = 0
        major_ct = 0
        significant_ct = 0
        minor_ct = 0
        method_ct = {'ts':0, 'sr':0, 'trend':0, 'seasonal':0, 'resid':0} 
        for dir_file in dir_list:
            key = dir_file.split(".")[0]
            if key in keys:
                ct += 1
                begin = []
                end = []  
                for an in dict_an[key] : 
                    begin.append(an[0])
                    end.append(an[1])
                print('Anomaly' + str(ct)+ ' (' + key + ')' + ':',begin,'-',end)

                value, best_detection = main(path + '/' + dir_file, False, begin, end, "NASA")
                val_ct += value
                if value >= 2.5:
                    major_ct += 1
                elif value >= 1.5:
                    significant_ct += 1
                elif value >= 0.5:
                    minor_ct += 1
                for method in best_detection:
                    method_ct[method] += 1 
                print('Value:', value)
                print('Detected by:', best_detection, '\n')
                f.write(str(ct) + ') ' + dir_file + ', ' + str(value) + ', ' + str(best_detection) + '\n')

        print('Result:', val_ct, '/', 3*ct)
        f.write('\n')
        f.write('Result: ' + str(val_ct) + '/' + str(3*ct))
        f.write('\n')
        f.write('Major: ' + str(major_ct) + ', Significant: ' + str(significant_ct) + ', Minor: ' + str(minor_ct))
        f.write('\n')
        for method in method_ct.keys():
            f.write(method + ': ' + str(method_ct[method]))
            f.write('    ')
        f.close()

    def example9():
        """UCR's anomaly database evaluation.

        Intent(in): None

        Returns: None
        """

        print("Example 9: UCR's anomaly database evaluation (highly time consuming).")

        #--- UCR anomalies ---
         
        path = os.getcwd() + '/UCR_Anomaly_FullData'
        dir_list = os.listdir(path)
        f = open('results.txt', 'w')
        f.write('UCR benchmark: \n')
        f.write('--- Major = 3, significant = 2, minor = 1, not found = 0--- \n')
        f.write('\n')
        val_ct = 0
        ct = 0
        major_ct = 0
        significant_ct = 0
        minor_ct = 0
        method_ct = {'ts':0, 'sr':0, 'trend':0, 'seasonal':0, 'resid':0} 
        for dir_file in dir_list:
            split = dir_file.split('_')
            if (int(split[0]) < 204 or int(split[0]) > 208) and (int(split[0]) < 242 or int(split[0]) > 243) and (int(split[0]) < 225 or int(split[0]) > 226):
                ct += 1
                begin = int(split[-2])
                end = int(split[-1].split('.')[0])  
                print('Anomaly' + str(ct)+':',begin,'-',end)

                value, best_detection = main(path + '/' + dir_file, False, [begin], [end], "UCR")
                val_ct += value
                if value >= 2.5:
                    major_ct += 1
                elif value >= 1.5:
                    significant_ct += 1
                elif value >= 0.5:
                    minor_ct += 1
                for method in best_detection:
                    method_ct[method] += 1 
                print('Value:', value)
                print('Detected by:', best_detection, '\n')
                f.write(str(ct) + ') ' + dir_file + ', ' + str(value) + ', ' + str(best_detection) + '\n')


        print('Result:', val_ct, '/', 3*ct)
        f.write('\n')
        f.write('Result: ' + str(val_ct) + '/' + str(3*ct))
        f.write('\n')
        f.write('Major: ' + str(major_ct) + ', Significant: ' + str(significant_ct) + ', Minor: ' + str(minor_ct))
        f.write('\n')
        for method in method_ct.keys():
            f.write(method + ': ' + str(method_ct[method]))
            f.write('    ')
        f.close()

    def example10():
        """Comparison between fortran and python algorithm for Mean Value Filter.

        Intent(in): None

        Returns: None
        """

        print("Example 10: Comparison between computational time for python and fortran filter.")

        [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/170_UCR_Anomaly_gaitHunt1_18500_33070_33180.txt")

        plot(t,X)

        N = array([i for i in range(0,500)])
        M = len(X)
        Y = zeros(M)

        t0 = 0
        tf = 0
        t_fortran = []
        t_python = []  

        alpha = 1
        a1 = 1./(2*(alpha+1))
        a0 = alpha/(alpha+1)

        for j in N:
            #fortran filter 
            t0 = time.time()
            Y = fortran_ts.time_series.mvf(asfortranarray(X), alpha)
            tf = time.time()
            t_fortran.append(tf-t0)

            #python filter
            t0 = time.time()
            for i in range(1,M-1):
                Y[i] = a1*X[i-1] + a0*X[i] + a1*X[i+1]
            tf = time.time()
            t_python.append(tf-t0)

        for i in range(1,len(N)):
            t_fortran[i] = t_fortran[i] + t_fortran[i-1]   
            t_python[i] = t_python[i] + t_python[i-1]  

        t_fortran = array(t_fortran)
        t_python = array(t_python) 

        plt.figure()
        plt.plot(N,t_fortran,'b')
        plt.plot(N,t_python,'r')
        plt.title('Comparison between computational time for N iterations of the filter')
        plt.xlabel('N iterations')
        plt.ylabel('t (s)', rotation=0)
        plt.legend(['Fortran', 'Python'])

        plt.figure()
        plt.plot(N,t_python/t_fortran,'g')
        plt.title('Quotient of python and fortran time for N iterations of the filter')
        plt.xlabel('N iterations')
        plt.ylabel('r', rotation=0)
        plt.legend(['t_Python / t_Fortran'])

        plt.show()

    def example_invalid():
        print('Invalid case selected. Select an example from 1 to 10.')


    #Switch case dictionary 
    switcher = {1: example1, 2:example2, 3:example3, 4:example4, 5:example5, 6:example6, 7:example7, 8:example8, 9:example9, 10:example10}
    #Get the function from switcher dictionary  
    example = switcher.get(N, example_invalid)

    return example()


if __name__ == "__main__":

    run = True
    while run:
        print(""" Select an introductory pre-defined example:\n 
        0) Exit\n 
        1) Noise reduction.\n 
        2) Period estimation.\n 
        3) Time series decomposition: STL.\n 
        4) Time series decomposition: MVD.\n 
        5) Anomaly detection with spectral residual transformation.\n 
        6) Anomaly detection with ADTK.\n 
        7) Time series decomposition (STL + MVD), spectral residual transformation\n \t and anomaly detection with ADTK.\n 
        8) NASA's anomaly database evaluation (highly time consuming).\n 
        9) UCR's anomaly database evaluation (highly time consuming).\n 
        10) Comparison between computational time for python and fortran filter.\n 
        """)

        option = input("Select an example from 0 to 10: ")
        if option == '0':
            run = False
        else:
            user_examples(int(option))
        run = False






    # value, best_detection = main("UCR_Anomaly_FullData/146_UCR_Anomaly_Lab2Cmac011215EPG2_5000_27862_27932.txt", True, 27862, 27932)
    # value, best_detection = main("UCR_Anomaly_FullData/098_UCR_Anomaly_NOISEInternalBleeding16_1200_4187_4199.txt", True, 4150, 4199)
    # value, best_detection = main("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt", True, [5988] ,[6085])
    # value, best_detection = main("test_NASA/G-1.npy", True,[4770] ,[4890])
    # value, best_detection = main("UCR_Anomaly_FullData/250_UCR_Anomaly_weallwalk_2951_7290_7296.txt", True, [7290], [7296])

    # print(value, best_detection)

    # [t, X] = test_function.solar_power_sso(1) 
    # [t, X] = test_function.sin_function() 
    # [t, X] = test_function.square_function() 
    # [t, X] = test_function.cubic_function() 
    # [t, X] = test_function.test_sine()
    # [t, X] = test_function.read("20211014.plt") 
    # [t, X] = test_function.load_npy("P-11.npy") 
    # [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")