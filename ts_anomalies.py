from adtk.detector import ThresholdAD, LevelShiftAD, VolatilityShiftAD, QuantileAD
from adtk.visualization import plot
from adtk.data import validate_series

from numpy import mean, std, array, logical_and, logical_or, logical_xor, zeros, where, angle, log, exp, convolve, ones
from scipy.fft import fft, fftfreq, ifft

class Anomaly_detection():
    def __init__(self, df, labels, plot_anomalies=True):
        self.plot_anomalies = plot_anomalies
        self.anomaly_tag ={'th':'marker', 'ls':'span', 'vol':'span'} 
        self.anomaly_color ={'th':'red', 'ls':'orange', 'vol':'green'} 
        self.master_dict = {} 
        ct = 0
        # aux0 = array([False for _ in range(0,len(df['X(t)']))])
        # aux1 = array([False for _ in range(0,len(df['X(t)']))]) 
        # aux2 = array([False for _ in range(0,len(df['X(t)']))])
        # aux3 = array([False for _ in range(0,len(df['X(t)']))])
        ct_anomaly = zeros(len(df['X(t)']))
        for tag in labels:
            if tag == "ts":
                ct += 1
                self.analysis = ['th', 'ls', 'vol'] 
                self.sigma_th = 3
                self.c_ls = 12.0
                self.c_vol = 20.0
                self.ts_dict = self.detector(df['X(t)'])
                ct_anomaly = zeros(len(self.ts_dict['th']))
                for key in self.analysis:
                    # aux0  = logical_or(aux0, self.ts_dict[key])
                    for i in range(int(len(ct_anomaly)/20),int(19*len(ct_anomaly)/20)):
                        if self.ts_dict[key][i] == True:
                            ct_anomaly[i] += 1
                            if key == 'th':
                                ct_anomaly[i] += 1

            if tag == "sr":
                ct += 1
                self.analysis = ['th', 'ls', 'vol'] 
                self.sigma_th = 3
                self.c_ls = 12.0
                self.c_vol = 20.0
                self.sr_dict = self.detector(df['sr'])
                for key in self.analysis:
                    # aux0  = logical_or(aux0, self.ts_dict[key])
                    for i in range(int(len(ct_anomaly)/20),int(19*len(ct_anomaly)/20)):
                        if self.sr_dict[key][i] == True:
                            ct_anomaly[i] += 1

            if tag == "trend":
                ct += 1
                self.analysis = ['th', 'ls'] 
                self.sigma_th = 3
                self.c_ls = 12.0
                self.trend_dict = self.detector(df['trend'])
                for key in self.analysis:
                    # aux1  = logical_or(aux1, self.trend_dict[key])
                    for i in range(int(len(ct_anomaly)/20),int(19*len(ct_anomaly)/20)):
                        if self.trend_dict[key][i] == True:
                            ct_anomaly[i] += 2

            if tag == "seasonal":
                ct += 1
                self.analysis = ['th', 'ls'] 
                self.sigma_th = 3
                self.c_ls = 8.0
                self.seasonal_dict = self.detector(df['seasonal'])

                for key in self.analysis:
                    # aux2 = logical_or(aux2, self.seasonal_dict[key])
                    for i in range(int(len(ct_anomaly)/20),int(19*len(ct_anomaly)/20)):
                        if self.seasonal_dict[key][i] == True:
                            ct_anomaly[i] += 2

            if tag == "resid":
                ct += 1
                self.analysis = ['th', 'vol'] 
                self.sigma_th = 4
                self.c_vol = 18.0
                self.resid_dict = self.detector(df['resid'])

                for key in self.analysis:
                    # aux3 = logical_or(aux3, self.resid_dict[key]) 
                    for i in range(int(len(ct_anomaly)/20),int(19*len(ct_anomaly)/20)):
                        if self.resid_dict[key][i] == True:
                            ct_anomaly[i] += 1

            if ct >= 4:
                # logical_aux = logical_or(logical_or(logical_and(aux2, aux3), logical_and(aux0, aux2)), logical_and(aux0, aux3))
                # self.master_dict['major']  = list(logical_or(aux1, logical_aux))
                # self.master_dict['minor']  = list(logical_or(aux1, logical_or(aux2, aux3)))

                #Major anomaly 
                if max(ct_anomaly) > 0:
                    ct_max_loc = where(ct_anomaly == max(ct_anomaly))
                    self.master_dict['major'] = zeros(len(ct_anomaly))
                    for item in ct_max_loc:
                        self.master_dict['major'][item]  =  1
                        ct_anomaly[item] = 0 

                #Significant anomaly
                if max(ct_anomaly) > 0:
                    self.master_dict['significant'] = zeros(len(ct_anomaly))
                    ct_max_loc = where(ct_anomaly == max(ct_anomaly))
                    for item in ct_max_loc:
                        self.master_dict['significant'][item]  =  1
                        ct_anomaly[item] = 0 

                #Minor anomaly
                if max(ct_anomaly) > 0:
                    self.master_dict['minor'] = zeros(len(ct_anomaly))
                    while max(ct_anomaly) > 1:
                        ct_max_loc = where(ct_anomaly == max(ct_anomaly))
                        for item in ct_max_loc:
                            self.master_dict['minor'][item]  =  1
                            ct_anomaly[item] = 0

            # tag ={'major':'span', 'minor':'span'} 
            # color ={'major':'red', 'minor':'green'} 
            
            # plot(df['X(t)'], anomaly=self.master_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=color, anomaly_tag=tag)
                

    def detector(self,X):
        ts = validate_series(X)
        ts_dict = {}  
        #Threshold 
        if 'th' in self.analysis:
            sigma_ts = std(array(X))
            mean_ts = mean(array(X))
            threshold_ad = ThresholdAD(high=mean_ts + self.sigma_th*sigma_ts, low=mean_ts - self.sigma_th*sigma_ts)
            th_anomalies = threshold_ad.detect(ts)
            ts_dict['th'] = th_anomalies 
        #Level shift 
        if 'ls' in self.analysis:
            level_shift_ad = LevelShiftAD(c=self.c_ls, side='both', window=20)
            ls_anomalies = level_shift_ad.fit_detect(ts)
            ts_dict['ls'] = ls_anomalies 
        #Volatility 
        if 'vol' in self.analysis:
            volatility_shift_ad = VolatilityShiftAD(c=self.c_vol, side='both', window=30)
            if max(abs(ts)) > 0:
                vol_anomalies = volatility_shift_ad.fit_detect(ts)
                ts_dict['vol'] = vol_anomalies
            else:
                self.analysis.remove('vol')

        tag = {}
        color = {}
        for key in ts_dict.keys():
            color[key]  = self.anomaly_color[key]
            tag[key]  = self.anomaly_tag[key]
        if self.plot_anomalies:
            plot(ts, anomaly=ts_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=color, anomaly_tag=tag)

        return ts_dict

    

if __name__ == "__main__":
    import test_function
    from matplotlib import pyplot as plt
    [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/146_UCR_Anomaly_Lab2Cmac011215EPG2_5000_27862_27932.txt")
    # [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt") 

    # t = t[27000:29000] 
    # X = X[27000:29000] 

    for i in range(1,10):
        for j in range(1,len(X)-1):
            X[j] = (X[j-1] + 2*X[j] + X[j+1])/4    

    X_FFT = fft(X)
    A = abs(X_FFT)
    P = angle(X_FFT)
    L = log(A)
    q = 3
    hq = ones(q)/q
    R = L - convolve(L,hq, 'same')
    S = abs(ifft(exp(R + 1j*P))**2)

    plt.figure()
    plt.plot(t, S, 'b')
    # plt.plot(t, X, 'r')
    plt.show()

        
            