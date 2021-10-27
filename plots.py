from numpy import pi, cos, linspace
from matplotlib import pyplot as plt

def gain_periodic(t,alpha):
    G = (alpha + cos(2*pi*t))/(alpha+1)
    return G

if __name__ == "__main__":
    t = linspace(0,1,100)

    #Periodic gain
    plt.figure()
    legend =[] 
    plt.plot(t, 0*t, 'r--')
    legend.append('Maximum dampening')
    for i in range(-1,4):
        plt.plot(t, gain_periodic(t,i/2.)) 
        legend.append(r'$\alpha$='+str(i/2.))
    plt.xlabel('$\it{f}$', fontsize = 18)
    plt.ylabel('$\it{G}$', fontsize = 18, rotation=0)
    plt.legend(legend, fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.show()