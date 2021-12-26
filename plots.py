import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq, ifft

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
    [t, X] = test_function.read_UCR("156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt")

    N_vec, E = period_est(t, X)

    plt.figure()
    plt.plot(t, X)
    plt.xlabel('$\it{t}$ (s)', fontsize = 18)
    plt.ylabel('$\it{X}$', fontsize = 18, rotation=0)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)

    plt.figure()
    plt.plot(N_vec, E)
    plt.xlabel('$\it{N}$', fontsize = 18)
    plt.ylabel('$\it{E}$', fontsize = 18, rotation=0)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.show()
