import warnings
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

def ts_decomposition(df,**kwargs):
    """Time series decomposition in trend, seasonal/cyclic variations
    and irregular variations.

        Intent(in): df(pandas DataFrame), time series;
        kwargs (optional): period(integer), time series period;
        plot(boolean), plot results, default = True

        Returns: decomposition(statsmodels class), time series decomposed in .trend, .seasonal and .resid
    """

    if "plot" in kwargs:
        plot = kwargs["plot"] 
        if type(plot) != bool:
            plot = True
            warnings.warn("plot argument must be of type bool, it has been set True by default.", stacklevel=2)
    else:
        plot = True

    X = np.array(df[df.columns[0]])
    t = np.array(df.index)

    # Analysis in frequency domain: FFT
    yf = fft(X)
    xf = fftfreq(len(t),t[1]-t[0])

    yf_max = np.max(np.abs(yf))
    yf_order =[]
    threshold = 0.05
    eps = 1e-5

    for i in range(0,len(X)//2):
        if abs(yf[i]) > eps:
            yf_order.append(i)
    
    yf_order = np.flip(np.array(yf_order))
    

    for i in yf_order:
        if abs(yf[i]) > threshold*yf_max:
            xf_th = xf[i]  
            break

    if plot:
        plt.figure()
        plt.plot(xf[:len(X)//2], 2/len(X) * np.abs(yf[0:len(X)//2]))
        plt.xlabel('f [Hz]')
        plt.ylabel('FFT')
        plt.title('FFT time series') 

    # Time series decomposition
    if "period" in kwargs:
        try:
            if type(kwargs["period"]) == int:
                period = kwargs["period"]
            else:
                period = int(kwargs["period"])
                warnings.warn("period argument must be of type integer, it has been rounded by default.", stacklevel=2)
        except:
            warnings.warn("period argument must be of type integer, it has been automatically computed.", stacklevel=2)
            period = int(1./(xf_th*(t[1] - t[0])))
    else:
        period = int(1./(xf_th*(t[1] - t[0])))

    print("period=", period, ", f=", xf_th)

    decomposition = seasonal_decompose(df[df.columns[0]], model="additive", period=period)

    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t,decomposition.trend)
        plt.xlabel('t')
        plt.title('Trend component') 
        plt.subplot(3,1,2)
        plt.plot(t,decomposition.seasonal)
        plt.xlabel('t')
        plt.title('Seasonal component') 
        plt.subplot(3,1,3)
        plt.plot(t,decomposition.resid)
        plt.xlabel('t')
        plt.title('Irregular variations') 

    return decomposition
