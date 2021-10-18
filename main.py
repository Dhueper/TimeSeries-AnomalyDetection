from numpy import array, transpose, zeros
import pandas as pd
from matplotlib import pyplot as plt
# from tsfresh import extract_features

import test_function
import ts_analysis

#%% Time series definition
# [t, X] = test_function.solar_power_sso(1) 
# [t, X] = test_function.sin_function() 
# [t, X] = test_function.square_function() 
# [t, X] = test_function.cubic_function() 
[t, X] = test_function.test_sine()

#Original time series plot
plt.figure()
plt.plot(t,X)
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title('Original time series')

id_column = zeros(len(t))
time_series = transpose(array([t,X,id_column]))

df = pd.DataFrame(time_series, columns=["time", "X(t)", "id"]) 

# Feature extraction with tsfresh 
# features = extract_features(df, column_id='id', column_sort='time')
# print(features)

df.set_index("time", inplace=True) 

#Decomposition of the time series. Available methods: 'STL', 'seasonal_decompose' and 'mean_value'
decomposition = ts_analysis.ts_decomposition(df, plot=True, method='mean_value', noise_filter=False)

plt.show()


