import numpy as np
import pandas as pd

import test_function

#%% Time series decomposition

[t, X] = test_function.solar_power_sso() 

time_serie = np.transpose(np.array([t,X]))

df = pd.DataFrame(time_serie, columns=["time", "X(t)"])

print(df)
