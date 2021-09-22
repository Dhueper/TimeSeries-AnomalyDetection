import numpy as np
import pandas as pd
from dateutil.parser import parse
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt

import test_function

#%% Time series decomposition

[t, X] = test_function.solar_power_sso() 

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

decomposition = seasonal_decompose(df['X(t)'], model="additive", period=5000)

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
