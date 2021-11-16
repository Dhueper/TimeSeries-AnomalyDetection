import warnings
from numpy import array, flip, zeros, var, append, float64, mean, sqrt, pi, cos, sin, matmul, log10, exp, linalg, diag
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy.fft import fft, fftfreq, ifft
from scipy.optimize import fsolve, minimize
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

import CNN_test

def ts_decomposition(df,**kwargs):
    """Time series decomposition in trend, seasonal/cyclic variations
    and irregular variations.

        Intent(in): df(pandas DataFrame), time series;
        kwargs (optional): period(integer), time series period;
        plot(boolean), plot results, default = True;
        method(string), method used for decomposition, 'seasonal_decompose', 'STL' (default), 'mean_value' or 'CNN';
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

    # X = array(df[df.columns[0]])
    X = array(df['X(t)'])
    # t = array(df.index)
    t = array(df['time'] )

    # Analysis in frequency domain: FFT
    X_FFT = Fourier(t,X)

    if X_FFT.f_th < 5/(len(X) * (t[1] - t[0])):
        X_FFT.f_th = 5/(len(X) * (t[1] - t[0]))

    if plot:
        plt.figure()
        plt.plot(X_FFT.f[:len(X)//2], 2/len(X) * abs(X_FFT.Xf[0:len(X)//2]), '+')
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
            period = round(1./(max(X_FFT.f_th,5/(len(X)*(t[1] - t[0]))) * (t[1] - t[0])))# period estimation 
            period = min(period,round(len(X)/10))
            print("period=", period, ", f=", X_FFT.f_th, " [Hz]")
    else:
        period = round(1./(max(X_FFT.f_th,5/(len(X)*(t[1] - t[0]))) * (t[1] - t[0])))# period estimation 
        period = min(period,round(len(X)/10))
        print("period=", period, ", f=", X_FFT.f_th, " [Hz]")


    #Noise reduction
    if "noise_filter" in kwargs:
        if kwargs["noise_filter"]:
            noise_filter = True
            n_noise_filter = max(round(period/10),1) #Times the recursive noise filter is applied 
            noiseless = Mean_value_decomposition(X, n_noise_filter, period, t, True, X_FFT.f_th, False)
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
            n_decom = int(100*period)
            decomposition = Mean_value_decomposition( X, n_decom, period, t, False, X_FFT.f_th, False)
        elif kwargs['method'] == 'CNN':
            n_decom = int(50*period)
            decomposition = Mean_value_decomposition( X, n_decom, period, t, False, X_FFT.f_th, True)
        else:
            warnings.warn("Unavailable method, used STL by default.", stacklevel=2)
            decomposition = STL(X, period=period).fit()
    else:
        decomposition = STL(X, period=period).fit()

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
            decomposition.resid[:] = decomposition.resid[:] + noiseless.resid[:]   

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

    def __init__(self, X, n, period, t, noise_filter, f_th, CNN_filter):
        self.M = len(X)
        self.t = t
        self.period = period
        self.trend = zeros(self.M)
        self.seasonal = zeros(self.M)
        self.resid = zeros(self.M)

        #Trend component 
        self.trend[:] = X[:]

        if CNN_filter:
            model = load_model('CNN_filter.h5')
            X_CNN = CNN_test.adjust_shape(X)
            X_CNN, Y_CNN = CNN_test.make_timeseries_instances(X_CNN, X_CNN, 50)
            self.trend = model.predict(X_CNN).reshape(-1)

        if noise_filter: # If it is a noise reduction operation
            for _ in range(0,n):
                self.trend = self.mean_value_filter(self.trend, False) 

        else: # If not a noise reduction operation  

            #Equal variance algorithm for BC 
            for _ in range(0,int(n/40)):
                self.trend = self.mean_value_filter(self.trend, True, alpha=1)
                self.trend = self.mean_value_filter(self.trend, True, alpha=2)


            #Linear interpolation to correct end-point desviations 
            k = min(4*period, int(len(X)/4))
            d_trend = zeros(k-int(k/2)+1)
            for i in range(int(k/2),k+1):
                d_trend[i-int(k/2)] = (self.trend[i+1] - self.trend[i-1])/(2)  
            for i in range(0,int(k/2)):
                self.trend[i] = mean(self.trend[1:1+k]) - mean(d_trend)*(int(k/2)-i)

            for i in range(int(k/2),k+1):
                d_trend[i-int(k/2)] = (self.trend[self.M-i] - self.trend[self.M-i-2])/(2) 
            for i in range(0,int(k/2)):
                self.trend[self.M-1-i] = mean(self.trend[self.M-2-k:self.M-2]) + mean(d_trend)*(int(k/2)-i)

            #Linear BC 
            alpha = max(0,-cos(2*pi*f_th*(t[1]-t[0])))
            print('alpha=',alpha)
            for i in range(0,int(9*n/10)):
                aux_trend = self.mean_value_filter(self.trend, False,alpha=alpha)
                # if abs(var(aux_trend, dtype=float64) - var(self.trend, dtype=float64))/var(aux_trend, dtype=float64) < 10**(-9 - int(log10(var(aux_trend, dtype=float64)))):
                if max(abs(aux_trend - self.trend))*var(aux_trend, dtype=float64) < (max(aux_trend)-min(aux_trend)) * 1e-7:
                    self.trend[:] = aux_trend[:] 
                    print('n_max=',i)
                    break
                else:
                    self.trend[:] = aux_trend[:] 

            self.seasonal[:] = X[:] - self.trend[:] #Detrended time series 
            if period > 20:
                for _ in range(0,3):
                    self.seasonal = self.mean_value_filter(self.seasonal, False)

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
                    self.resid = self.mean_value_filter(self.resid, False)

            resid_f = fft(self.resid)
            resid_fmax = max(abs(resid_f))
            resid_th = 0.005

            for i in range(0,len(resid_f)):
                if abs(resid_f[i]) < resid_fmax * resid_th:
                    resid_f[i] = 0

            self.seasonal = self.seasonal + ifft(resid_f).real #Seasonal component     

        self.resid[:] = X[:] - self.trend[:] - self.seasonal #Residual component 

        # Comparison of magnitude orders
        components = [self.trend, self.seasonal, self.resid] 
        max_components = [max(abs(self.trend)), max(abs(self.seasonal)), max(abs(self.resid))]
        max_mean =[] 

        for j in range(0,3):
            selected_index = (-abs(components[j])).argsort()[:int(self.M*0.1)]
            max_mean.append(mean(array([abs(components[j][i]) for i in selected_index])))

        max_index = max_mean.index(max(max_mean))

        for i in range(0,3):
            # if i != max_index and max_components[i] < 0.05*max_mean[max_index]:
            if i != max_index and max_components[i] < 0.1*max_mean[max_index]:
                components[max_index] = components[max_index] + components[i]
                components[i] = zeros(self.M)

        self.trend = components[0]
        self.seasonal = components[1]
        self.resid = components[2]     

    def mean_value_filter(self, X, trend, alpha=2):
        """Time series filter based on the mean value theorem and a discrete integration rule:
        alpha=1 (linear), alpha=2 (quadratic).

            X (numpy.array), time series;
            trend (bool), True if the trend is to be computed;
            alpha (integer), order of the filter.

            Returns: Y (numpy.array), filtered time series.
        """

        def f_var(x, Y, j): 
            """ Computes the difference between the variance and the second derivative with and without an end-point. 

            x (float), end point;
            Y (numpy.array), time series without end points;
            j (integer), index of array Y.

            Returns: delta_abs (float), difference between variances and second derivatives.
            """
            delta_var = var(Y, dtype=float64) - var(append(Y,x), dtype=float64)

            if j<2:
                delta_d2Y = (Y[j] - 2*Y[j+1] + Y[j+2]) - (x - 2*Y[j] + Y[j+1])
            else:
                k = int(j-len(Y))+1
                delta_d2Y = (Y[j-k] - 2*Y[j-k-1] + Y[j-k-2]) - (x - 2*Y[j-k] + Y[j-k-1])

            delta_abs = abs(delta_var) + abs(delta_d2Y)
            return delta_abs

        Y = zeros(self.M, dtype=float64)    

        for i in range(1,self.M-1):
            Y[i] = (X[i-1] + 2*alpha*X[i] + X[i+1])/(2. * (alpha+1)) 
        
        if trend: # Trend decomposition 

            Y[0] = minimize(f_var, x0=X[0], args=(Y[1:self.M-1], 0), tol=0.1*abs(X[0])).x

            Y[self.M-1] = minimize(f_var, x0=X[self.M-1], args=(Y[1:self.M-1], self.M-1), tol=0.1*abs(X[self.M-1])).x

            Y[1] = (Y[1] + minimize(f_var, x0=Y[1], args=(Y[4:self.M-4], 1), tol=0.1*abs(Y[1])).x)/2. 

            Y[self.M-2] = (Y[self.M-2] + minimize(f_var, x0=Y[self.M-2], args=(Y[4:self.M-4], self.M-2), tol=0.1*abs(Y[self.M-2])).x)/2. 
            
        else: # Noise filter or smoothing the seasonal component

            Y[0] = 2*Y[1] - Y[2] 
            Y[self.M-1] = 2*Y[self.M-2] - Y[self.M-3] 
            # Y[0] = (2*X[0] + X[1])/4.  
            # Y[self.M-1] = (2*X[self.M-1] + X[self.M-2])/4.
        
        return Y