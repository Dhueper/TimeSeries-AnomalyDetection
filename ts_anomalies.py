from adtk.detector import ThresholdAD, LevelShiftAD, VolatilityShiftAD
from adtk.visualization import plot
from adtk.data import validate_series

from numpy import mean, std, array, logical_and, logical_or, logical_xor

class Anomaly_detection():
    def __init__(self, df, labels):
        self.anomaly_tag ={'th':'marker', 'ls':'span', 'vol':'span'} 
        self.anomaly_color ={'th':'red', 'ls':'orange', 'vol':'green'} 
        self.master_dict = {} 
        ct = 0
        aux0 = array([False for _ in range(0,len(df['X(t)']))])
        aux1 = array([False for _ in range(0,len(df['X(t)']))]) 
        aux2 = array([False for _ in range(0,len(df['X(t)']))])
        aux3 = array([False for _ in range(0,len(df['X(t)']))])
        for tag in labels:
            if tag == "ts":
                ct += 1
                self.analysis = ['th', 'ls', 'vol'] 
                self.sigma_th = 3
                self.c_ls = 3.5
                self.c_vol = 18.0
                self.ts_dict = self.detector(df['X(t)'])
                for key in self.analysis:
                    aux0  = logical_or(aux0, self.ts_dict[key])

            if tag == "trend":
                ct += 1
                self.analysis = ['th', 'ls'] 
                self.sigma_th = 3
                self.c_ls = 4.0
                self.trend_dict = self.detector(df['trend'])
                for key in self.analysis:
                    aux1  = logical_or(aux1, self.trend_dict[key])

            if tag == "seasonal":
                ct += 1
                self.analysis = ['th', 'ls'] 
                self.sigma_th = 3
                self.c_ls = 3.5
                self.seasonal_dict = self.detector(df['seasonal'])

                for key in self.analysis:
                    aux2 = logical_or(aux2, self.seasonal_dict[key])

            if tag == "resid":
                ct += 1
                self.analysis = ['th', 'vol'] 
                self.sigma_th = 4
                self.c_vol = 6.0
                self.resid_dict = self.detector(df['resid'])

                for key in self.analysis:
                    aux3 = logical_or(aux3, self.resid_dict[key]) 

            if ct == 4:
                self.master_dict['major']  = list(logical_or(aux1, logical_and(aux0, logical_and(aux2, aux3))))
                self.master_dict['minor']  = list(logical_or(aux1, logical_or(aux2, aux3)))

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

        tag = {}
        color = {}
        for key in ts_dict.keys():
            color[key]  = self.anomaly_color[key]
            tag[key]  = self.anomaly_tag[key]

        plot(ts, anomaly=ts_dict, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color=color, anomaly_tag=tag)

        return ts_dict

        
            