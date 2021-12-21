import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, ifft

import test_function

# Ns = 1000
# t = np.linspace(0,2*np.pi,Ns, endpoint=False) 
# delta_t = t[1] - t[0]  

# xs = np.sin(16 * t) + np.sin(24 * t) + np.cos(12 * t) 
# print('period=', 2*np.pi/(24*delta_t))

# dxs = np.zeros(Ns)
# dxs[0] = (xs[1] - xs[0])/delta_t
# dxs[Ns-1] = (xs[Ns-1] - xs[Ns-2])/delta_t
# for i in range(1,Ns-1):
#     dxs[i] = (xs[i+1] - xs[i-1])/(2*delta_t) 

# N_vec = np.array([i for i in range(2,513)])
# E = [] 
# for N in N_vec: 
#     k = 1
#     aux = np.zeros(k)

#     for j in range(0,k):

#         #First N samples 
#         x = xs[N*j:N*(j+1)] 
#         dx = dxs[N*j:N*(j+1)] 

#         #Spectral domain
#         xf = fft(x, N)
#         f = fftfreq(N,delta_t)  

#         dx_spec = ifft(2*np.pi*f*1j*xf, N).real 

#         aux[j] =  np.sqrt(np.sum((dx - dx_spec)**2.) / N)

#     #Error 
#     E.append(np.sum(aux))
#     # print('Error RMS =', E)



# plt.figure()
# plt.plot(N_vec, E)
# plt.xlabel('N')
# plt.ylabel('E')
# plt.show()

#Plot results
# plt.figure()
# plt.plot(t[0:N] ,dx ,'b',label='Exact value')
# plt.plot(t[0:N] ,dx_spec,'r',label='Spectral derivative')
# plt.legend()

# plt.figure()
# plt.plot(t[0:N] , dx -dx_spec,'g')
# plt.title('Error')
# plt.show() 

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

    plt.figure()
    plt.plot(N_vec, E)
    plt.xlabel('N')
    plt.ylabel('E')
    plt.show()

    return N_vec[np.argmin(E)] 


if __name__ == "__main__":
    [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")
    # [t, X] = test_function.solar_power_sso(1)
    # Ns = 1000
    # t = np.linspace(0,2*np.pi,Ns, endpoint=False) 

    # X = np.sin(16 * t) + np.sin(24 * t) + np.cos(12 * t)

    print(t[1] - t[0])
    plt.figure()
    plt.plot(t,X)
    plt.show()
    period = period_est(t,X)
    print(period)