import warnings
from numpy import array, flip, zeros, poly1d, polyfit
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy.fft import fft, fftfreq, ifft
from scipy import interpolate
from matplotlib import pyplot as plt

def ts_decomposition(df,**kwargs):
    """Time series decomposition in trend, seasonal/cyclic variations
    and irregular variations.

        Intent(in): df(pandas DataFrame), time series;
        kwargs (optional): period(integer), time series period;
        plot(boolean), plot results, default = True;
        method(string), method used for decomposition, 'seasonal_decompose' (default) or 'STL';
        noise_filter(boolean), apply noise reduction through mean value decomposition, default=False.

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
    X_FFT = Fourier(t,X)

    if plot:
        plt.figure()
        plt.plot(X_FFT.f[:len(X)//2], 2/len(X) * abs(X_FFT.Xf[0:len(X)//2]))
        plt.xlabel('f [Hz]')
        plt.ylabel('FFT')
        plt.title('FFT time series') 

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
            period = round(1./(X_FFT.f_th*(t[1] - t[0])))# period estimation 
            print("period=", period, ", f=", X_FFT.f_th, " [Hz]")
    else:
        period = round(1./(X_FFT.f_th*(t[1] - t[0])))# period estimation 
        print("period=", period, ", f=", X_FFT.f_th, " [Hz]")


    #Noise reduction
    if "noise_filter" in kwargs:
        if kwargs["noise_filter"]:
            noise_filter = True
            n_noise_filter = max(round(period/10),1) #Times the recursive noise filter is applied 
            noiseless = Mean_value_decomposition(X, n_noise_filter, period, t)
            X[:] = noiseless.trend[:] + noiseless.seasonal[:]   
        else:
            noise_filter = False
    else:
        noise_filter = False


    if noise_filter and period < 8:
        warnings.warn("the seasonal period is very short and the noise filter may significantly modify the result."+
         " It is recommended to switch 'noise_filter' to False.", stacklevel=2)

         
    # Time series decomposition

    if "method" in kwargs:
        if kwargs['method'] == 'seasonal_decompose':
            decomposition = seasonal_decompose(X, model="additive", period=period)
        elif kwargs['method'] == 'STL':
            decomposition = STL(X, period=period).fit()
        elif kwargs['method'] == 'mean_value':
            # decomposition = Mean_value_decomposition( X, max(int(len(X)/2),100*period))
            # n_decom = int(50*X_FFT.Xf_th*period)
            n_decom = int(100*period)
            print(n_decom)
            decomposition = Mean_value_decomposition( X, n_decom, period, t)
        else:
            warnings.warn("Unavailable method, used seasonal_decompose by default.", stacklevel=2)
            decomposition = seasonal_decompose(X, model="additive", period=period)
    else:
        decomposition = seasonal_decompose(X, model="additive", period=period)

    if plot:
        plt.figure()
        plt.subplot(4,1,1)
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

        if noise_filter:
            plt.plot(t,noiseless.resid, linewidth=1)
            plt.title('Irregular variations + noise') 

    return decomposition


class Fourier():
    """Frequency domain analysis through FFT.
    
            Intent(in): 
            t (numpy.array), timestamps;
            X (numpy.array), time series.

            Attributes: Xf (FFT), f (frequencies), Xf_max (max(Xf)) and f_th (threshold frequency).
         """
    def __init__(self,t,X):
        self.Xf = fft(X)
        self.f = fftfreq(len(t),t[1]-t[0])

        self.Xf_max = max(abs(self.Xf))
        Xf_order =[]
        threshold = 0.05
        eps = 1e-5

        #Relevant frequencies index
        for i in range(0,len(X)//2):
            if abs(self.Xf[i]) > eps:
                Xf_order.append(i)
        
        Xf_order = flip(array(Xf_order))
        
        #Last significant high frequency  
        for i in Xf_order:
            if abs(self.Xf[i]) > threshold*self.Xf_max:
                self.f_th = self.f[i]  
                self.Xf_th = 2/len(X) * abs(self.Xf[i])
                break

class Mean_value_decomposition():
    """Time series decomposition through n recurrent mean value filters.
    
            Intent(in): 
            X (numpy.array), time series;
            n (integer), times the recursive filter is applied.

            Attributes: trend, seasonal and resid.
         """

    def __init__(self, X, n, period, t):
        self.M = len(X)
        self.t = t
        self.poly_deg = 2
        self.trend = zeros(self.M)
        self.seasonal = zeros(self.M)
        self.resid = zeros(self.M)

        self.trend[:] = X[:]  

        #Trend component 
        for _ in range(0,n):
            self.trend = self.quadratic_mean_value_filter(self.trend) 

        if n > 10: # If not a noise reduction operation 
            self.seasonal[:] = X[:] - self.trend[:] #Detrended time series 
            if period > 20:
                for _ in range(0,3):
                    self.seasonal = self.quadratic_mean_value_filter(self.seasonal)

            seasonal_f = fft(self.seasonal)
            seasonal_fmax = max(abs(seasonal_f))
            seasonal_th = 0.02

            for i in range(0,len(seasonal_f)):
                if abs(seasonal_f[i]) < seasonal_fmax * seasonal_th:
                    seasonal_f[i] = 0

            self.seasonal = ifft(seasonal_f).real 


            self.resid[:] = X[:] - self.trend[:] - self.seasonal[:] #Detrended and deseasonalized time series
            if period > 20:
                for _ in range(0,5):
                    self.resid = self.quadratic_mean_value_filter(self.resid)

            resid_f = fft(self.resid)
            resid_fmax = max(abs(resid_f))
            resid_th = 0.005

            for i in range(0,len(resid_f)):
                if abs(resid_f[i]) < resid_fmax * resid_th:
                    resid_f[i] = 0

            self.seasonal = self.seasonal + ifft(resid_f).real #Seasonal component     
        
        # if n >=100:
        #     self.seasonal[:] = X[:] - self.trend[:] 
        #     for _ in range(0,int(n/200)):
        #         self.seasonal = self.mean_value_filter(self.seasonal)

        self.resid[:] = X[:] - self.trend[:] - self.seasonal #Residual component 


    def mean_value_filter(self, X):
        """Time series filter based on the mean value theorem and the trapezoid integration rule.

            X (numpy.array), time series.

            Returns: Y(numpy.array), filtered time series.
        """

        M = self.M
        Y = zeros(M)
        # Y[0] = X[0]  
        # Y[M-1] = X[M-1] 

        for i in range(1,M-1):
            Y[i] = (X[i-1] + 2*X[i] + X[i+1])/4. 

        Y[0] = (Y[1] + X[0])/2. 
        Y[M-1] = (Y[M-2] + X[M-1])/2.  
        
        return Y

    def quadratic_mean_value_filter(self, X):
        """Time series filter based on the mean value theorem and the simpson (quadratic) integration rule.

            X (numpy.array), time series.

            Returns: Y(numpy.array), filtered time series.
        """

        Y = zeros(self.M)

        #Extrapolate time series
        # p_0 = poly1d(polyfit(self.t[0:self.poly_deg+1], X[0:self.poly_deg+1],self.poly_deg))
        # p_f = poly1d(polyfit(self.t[self.M-self.poly_deg-1:self.M], X[self.M-self.poly_deg-1:self.M],self.poly_deg))

        p_0 = interpolate.interp1d(self.t[0:self.poly_deg+1], X[0:self.poly_deg+1], fill_value="extrapolate")
        p_f = interpolate.interp1d(self.t[self.M-self.poly_deg-1:self.M], X[self.M-self.poly_deg-1:self.M], fill_value="extrapolate")

        Y[0] = (p_0(self.t[0]-self.t[1]) + 4*X[0] + X[1])/6.
        Y[self.M-1] = (p_f(2*self.t[self.M-1]-self.t[self.M-2]) + 4*X[self.M-1] + X[self.M-2])/6.

        for i in range(1,self.M-1):
            Y[i] = (X[i-1] + 4*X[i] + X[i+1])/6. 

        # Y[0] = (Y[1] + X[0])/2. 
        # Y[self.M-1] = (Y[self.M-2] + X[self.M-1])/2.
        
        return Y