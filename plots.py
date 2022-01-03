import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from adtk.transformer import DoubleRollingAggregate
from adtk.visualization import plot
import pandas as pd

import test_function

def gain_periodic(t,alpha):
    G = (alpha + np.cos(2*np.pi*t))/(alpha+1)
    return G

def period_est(t, xs):
    Ns = len(t)
    delta_t = t[1] - t[0]  

    dxs = np.zeros(Ns)
    dxs[0] = (xs[1] - xs[0])/delta_t
    dxs[Ns-1] = (xs[Ns-1] - xs[Ns-2])/delta_t
    for i in range(1,Ns-1):
        dxs[i] = (xs[i+1] - xs[i-1])/(2*delta_t) 

    N_vec = np.array([i for i in range(8,Ns//10)])
    E = [] 
    for N in N_vec: 

        #First N samples 
        x = xs[0:N] 
        dx = dxs[0:N] 

        #Spectral domain
        xf = fft(x, N)
        f = fftfreq(N,delta_t)  

        dx_spec = ifft(2*np.pi*f*1j*xf, N).real 

        #Error 
        E.append(np.sum(np.sqrt(np.sum((dx - dx_spec)**2.) / N)))      

    return N_vec, E

def spectral_residual(X, c):
    X_FFT = fft(X)
    A = abs(X_FFT) + 1e-8
    P = np.angle(X_FFT)
    L = np.log(A)
    q = 3
    hq = np.ones(q)/q
    R = L - np.convolve(L,hq, 'same')
    S = abs(ifft(np.exp(R + 1j*P))**c)

    return S

if __name__ == "__main__":
    
    #Periodic gain
    # t = np.linspace(0,1,100)
    # plt.figure()
    # legend =[] 
    # plt.plot(t, 0*t, 'r--')
    # legend.append('Maximum dampening')
    # for i in range(0,4):
    #     plt.plot(t, gain_periodic(t,i/2.)) 
    #     legend.append(r'$\alpha$='+str(i/2.))
    # plt.xlabel('$\it{f}$', fontsize = 18)
    # plt.ylabel('$\it{G}$', fontsize = 18, rotation=0)
    # plt.legend(legend, fontsize = 18)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()

    # G0 = gain_periodic(t,1)
    # plt.figure()
    # legend =[] 
    # plt.plot(t, 0*t, 'r--')
    # legend.append('Maximum dampening')
    # for i in range(1,6):
    #     plt.plot(t, G0**i) 
    #     legend.append('$\it{N}$='+str(i))
    # plt.xlabel('$\it{f}$', fontsize = 18)
    # plt.ylabel('$\it{G^N}$', fontsize = 18, rotation=0)
    # plt.legend(legend, fontsize = 18, loc = 'lower left')
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()

    #Period estimation
    # [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")

    # N_vec, E = period_est(t, X)

    # plt.figure()
    # plt.plot(t, X)
    # plt.xlabel('$\it{t}$ (s)', fontsize = 18)
    # plt.ylabel('$\it{X}$', fontsize = 18, rotation=0)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)

    # plt.figure()
    # plt.plot(N_vec, E)
    # plt.xlabel('$\it{N}$', fontsize = 18)
    # plt.ylabel('$\it{E}$', fontsize = 18, rotation=0)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()

    #Anomalies
    # [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/185_UCR_Anomaly_resperation11_58000_110800_110801.txt")
    # [t, X] = test_function.load_npy("test_NASA/A-2.npy") 

    # plt.figure()
    # plt.plot(t, X)
    # plt.xlabel('$\it{t}$ (s)', fontsize = 18)
    # plt.ylabel('$\it{X}$', fontsize = 18, rotation=0)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()

    #Transformers 
    #--- Spectral residual --- 
    # [t, X] = test_function.read_UCR("UCR_Anomaly_FullData/185_UCR_Anomaly_resperation11_58000_110800_110801.txt")
    # X = X[108000:114000]
    # t = t[108000:114000]  

    # SR = spectral_residual(X,2)

    # plt.figure()
    # plt.plot(t[3:-3] , SR[3:-3] )
    # plt.xlabel('$\it{t}$ (s)', fontsize = 18)
    # plt.ylabel('$\it{SR}$', fontsize = 18, rotation=0)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()

    #--- Double rolling aggregate (median) --- 
    # [t, X] = test_function.load_npy("test_NASA/P-15.npy") 

    # datetime = pd.to_datetime(list(t), unit="s")

    # time_series = np.transpose(np.array([t,X]))

    # df = pd.DataFrame(time_series, columns=["time", "X(t)"]) 

    # df["datetime"] = datetime 

    # df.set_index("datetime", inplace=True) 

    # s_transformed = DoubleRollingAggregate(
    # agg="median",
    # window=5,
    # diff="diff").transform(df['X(t)'])

    # plt.figure()
    # plt.plot(t, s_transformed)
    # plt.xlabel('$\it{t}$ (s)', fontsize = 18)
    # plt.ylabel('$\it{Double Rolling Aggregate (median)}$', fontsize = 18, rotation=90)
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.show()

    #--- Double rolling aggregate (quantile) ---
    [t, X] = test_function.load_npy("test_NASA/P-3.npy") 

    datetime = pd.to_datetime(list(t), unit="s")

    time_series = np.transpose(np.array([t,X]))

    df = pd.DataFrame(time_series, columns=["time", "X(t)"]) 

    df["datetime"] = datetime 

    df.set_index("datetime", inplace=True) 

    s_transformed = DoubleRollingAggregate(
    agg="quantile",
    agg_params={"q": [0.1, 0.5, 0.9]},
    window=250,
    diff="l2").transform(df['X(t)'])

    plt.figure()
    plt.plot(t, s_transformed)
    plt.xlabel('$\it{t}$ (s)', fontsize = 18)
    plt.ylabel('$\it{Double Rolling Aggregate (quantile)}$', fontsize = 18, rotation=90)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.show() 