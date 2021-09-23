import numpy as np
import pandas as pd
from dateutil.parser import parse
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

import test_function

#%% Time series definition

[t, X] = test_function.solar_power_sso(1) 

time_series = np.transpose(np.array([t,X]))


df = pd.DataFrame(time_series, columns=["time", "X(t)"]) 

df.set_index("time", inplace=True)

print(df)

plt.figure()
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Original time series') 
plt.show()

#%% FFT
yf = fft(X)
xf = fftfreq(len(t),t[1] - t[0])

yf_order = np.flip(np.sort(np.abs(yf)))

yf_argorder = np.flip(np.argsort(np.abs(yf)))

yf2 = np.array([abs(yf[i]) for i in yf_argorder])

print(yf_order)
print(yf2)

plt.figure()
plt.plot(xf, np.abs(yf))
plt.xlabel('f [Hz]')
plt.ylabel('FF')
plt.title('FFT time series') 
plt.show()

#%% Time series decomposition
period = int(2*np.pi/(0.05*(t[1] - t[0])))
periodf = int(1./(np.max(xf)*(t[1] - t[0])))
print(period/periodf)

decomposition = seasonal_decompose(df['X(t)'], model="additive", period=periodf)

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
plt.show()

# %%
