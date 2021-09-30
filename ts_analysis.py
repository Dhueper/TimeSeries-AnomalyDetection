import warnings
from numpy import array, flip
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

def ts_decomposition(df,**kwargs):
    """Time series decomposition in trend, seasonal/cyclic variations
    and irregular variations.

        Intent(in): df(pandas DataFrame), time series;
        kwargs (optional): period(integer), time series period;
        plot(boolean), plot results, default = True;
        method(string), method used for decomposition, 'seasonal_decompose' (default) or 'STL'

        Returns: decomposition(statsmodels class), time series decomposed in .trend, .seasonal and .resid
    """


    if "plot" in kwargs:
        plot = kwargs["plot"] 
        if type(plot) != bool:
            plot = True
            warnings.warn("plot argument must be of type bool, it has been set True by default.", stacklevel=2)
    else:
        plot = True

    X = array(df[df.columns[0]])
    t = array(df.index)

    # Analysis in frequency domain: FFT
    Xf = fft(X)
    f = fftfreq(len(t),t[1]-t[0])

    Xf_max = max(abs(Xf))
    Xf_order =[]
    threshold = 0.05
    eps = 1e-5

    #Relevant frequencies index
    for i in range(0,len(X)//2):
        if abs(Xf[i]) > eps:
            Xf_order.append(i)
    
    Xf_order = flip(array(Xf_order))
    
    #Last significant high frequency  
    for i in Xf_order:
        if abs(Xf[i]) > threshold*Xf_max:
            f_th = f[i]  
            break

    if plot:
        plt.figure()
        plt.plot(f[:len(X)//2], 2/len(X) * abs(Xf[0:len(X)//2]))
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
            print("period=", period)
        except:
            warnings.warn("period argument must be of type integer, it has been automatically computed.", stacklevel=2)
            period = round(1./(f_th*(t[1] - t[0])))# period estimation 
            print("period=", period, ", f=", f_th, " [Hz]")
    else:
        period = round(1./(f_th*(t[1] - t[0])))# period estimation 
        print("period=", period, ", f=", f_th, " [Hz]")

    if "method" in kwargs:
        if kwargs['method'] == 'seasonal_decompose':
            decomposition = seasonal_decompose(df[df.columns[0]], model="additive", period=period)
        elif kwargs['method'] == 'STL':
            decomposition = STL(df[df.columns[0]], period=period).fit()
        else:
            warnings.warn("Unavailable method, used seasonal_decompose by default.", stacklevel=2)
            decomposition = seasonal_decompose(df[df.columns[0]], model="additive", period=period)
    else:
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

        plt.figure()
        plt.plot(t, X-decomposition.trend-decomposition.seasonal-decomposition.resid)
        plt.xlabel('t')
        plt.title('Error: X(t) - T(t) - S(t) - e(t)') 

    return decomposition
