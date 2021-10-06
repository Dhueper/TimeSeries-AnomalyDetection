from numpy import pi, arccos, arcsin, sin, cos, sqrt, linspace, zeros, array, random
from matplotlib import pyplot as plt 

def solar_power_sso(periods):
    """ Example of time series: 
    Power generated by four solar arrays 
    in a Sun-synchronous orbit. 
    The satellite is spinning around its
    Nadir axis and there are solar arrays 
    on the four sides. 
    
    Intent(in): periods(integer), number of orbital periods to compute
    Returns: t(np.array), time; Pt(np.array), total power
    """

    def inc_sso(a):
        #Intent in: a(float), semimajor axis.
        #Returns: i(float), inclination
         
        J2 = 1.0827*10**(-3.)
        d_alfa = 2*pi/(365.25*24*3600)# rad/s
        mu = 3.986044418*10**5.
        Rt = 6378# km
        i = arccos(-(2./3)*d_alfa/(J2*Rt**2.)*sqrt(a**7./mu))
        return i

    # Orbit definition
    Rt = 6378# km
    mu = 3.986044418*10**5.
    h = 450# km
    a = Rt+h# km
    T = 2*pi*sqrt(a**3./mu)# s
    n = 2*pi/T# rad/s
    # SSO orbit's inclination 
    inc = inc_sso(a)
    #Time for RAAN=0 
    H0 = 12# High-noon
    delta=15*(H0-12)*pi/180.
    #Solar angle  
    betha = arcsin(sin(delta)*sin(inc))

    #Time when eclipse begins
    rho = arcsin(Rt/a)
    phi_2 = arccos(cos(rho)/cos(betha))
    alfa_ecl = pi-phi_2
    t_ecl = alfa_ecl/n

    #Power generated by the solar array 
    G = 1360# W/m²
    f_oc = 0.867# Occupation's factor
    eta = 0.29# Efficiency
    A = 0.03# m^2
    omega = 0.05# rad/s
    N_period = periods
    Nt=6000*N_period
    t=linspace(0,N_period*T,Nt)
    Px = zeros((4,Nt))# W (4 faces N times)
    Py = zeros((4,Nt))# W (4 faces N times)
    P0 = zeros(Nt)
    P0 = G*A*f_oc*eta*(1-0.05*t/(3600*24))

    for i in range(0,4):
        Px[i,:] = [max(P0[j]*cos(betha)*cos(omega*t[j]+(i-1)*pi/2.)*sin(n*t[j]),0) for j in range(0,len(t))] 
        Py[i,:] = [max(P0[j]*sin(betha)*cos(omega*t[j]+(i-1)*pi/2.),0) for j in range(0,len(t))] 

    for j in range(0,Nt):
        for k in range(0,N_period):
            if t[j] > (T*k + t_ecl) and t[j] < (T*(k+1)-t_ecl):
                Px[:,j] = 0
                Py[:,j] = 0

    P_t = zeros(Nt)
    for i in range(0,4):
        P_t = P_t + Px[i,:] + Py[i,:] 

    P_t = P_t + 0.2*random.normal(0,1,len(t))

    return [t,P_t]  


def sin_function():
    t = linspace(0,1,1000)
    e = 0.2*random.normal(0,1,len(t))
    y = 2*sin(2*pi*10 * t) 
    # print(len(t)/(50 * (t[1]-t[0])))
    return [t,y]  

def square_function():
    t = linspace(0,100,100)
    y = zeros(len(t))
    for i in range(0,int(len(t)/2)):
        if i%2 == 1:
            y[2*i] = 1
            y[2*i+1] = 1
        else:
            y[2*i] = -1 
            y[2*i+1] = -1 

    return [t,y] 

if __name__ == "__main__":
    [t, P_t] = solar_power_sso(1)
     #Plots
    plt.figure()
    plt.plot(t,P_t)
    plt.xlabel('t')
    plt.ylabel('Pt')
    plt.title('Total power in one orbit') 
    plt.show()