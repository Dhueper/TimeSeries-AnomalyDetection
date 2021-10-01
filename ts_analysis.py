import warnings
from numpy import array, flip, zeros
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
        elif kwargs['method'] == 'mean_value':
            decomposition = Mean_value_decomposition(df[df.columns[0]], 1000)
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


class Mean_value_decomposition():
    """Time series decomposition through n recurrent mean value filters. """

    def __init__(self, X, n):
        """Initialization of the decomposition procedure.

            Intent(in): self (object);
            X (numpy.array), time series;
            n (integer), times the recursive filter is applied.

            Attributes: trend, seasonal and resid.
        """
        self.trend = zeros(len(X))
        self.seasonal = zeros(len(X))
        self.resid = zeros(len(X))

        self.trend[:] = X[:]  

        for _ in range(0,n):
            self.trend = self.mean_value_filter(self.trend)

        self.resid[:] = X[:] - self.trend[:]   



    def mean_value_filter(self, X):
        """Time series filter based on the mean value theorem and the trapezoid integration rule.

            Intent(in): self (object);
            X (numpy.array), time series.

            Returns: Y(numpy.array), filtered time series.
        """

        M = len(X)
        Y = zeros(M)
        # Y[0] = X[0]
        # Y[M-1] = X[M-1]  

        for i in range(1,M-1):
            Y[i] = (X[i-1] + 2*X[i] + X[i+1])/4. 

        Y[0] = Y[1] 
        Y[M-1] = Y[M-2] 
        
        return Y