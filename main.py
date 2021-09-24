import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import test_function
import ts_analysis

#%% Time series definition
[t, X] = test_function.solar_power_sso(1) 
# [t, X] = test_function.sin_function() 

time_series = np.transpose(np.array([t,X]))

df = pd.DataFrame(time_series, columns=["time", "X(t)"]) 
df.set_index("time", inplace=True)

plt.figure()
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Original time series') 
plt.show()

decomposition = ts_analysis.ts_decomposition(df, plot=True)

