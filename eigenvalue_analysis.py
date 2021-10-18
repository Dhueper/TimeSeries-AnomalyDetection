from numpy import linalg, linspace, pi, cos, zeros, sort, argsort, array
from matplotlib import pyplot as plt

#Analytical eigenvalues of a tridiagonal matrix
def analytical_ev(alpha):
    N = 100
    a = alpha/(alpha+1)
    b = 1./(2*(alpha+1))

    k = linspace(1,N,N)

    ev = a + 2*b*cos(k*pi/(N+1))

    plt.plot(k, ev)

def numerical_eigenvalues(A):
    pass


if __name__ == "__main__":
    #Analytical 
    l =[] 
    plt.figure()
    for alpha in range(1,4):
        analytical_ev(alpha)
        l.append('alpha='+str(alpha))
    analytical_ev(0.5)
    l.append('moving averages')
    plt.legend(l)
    plt.xlabel('t')
    plt.ylabel('eigenvalues')
    plt.title('Analytical eigenvalues')

    #Numerical
    N = 100
    alpha = 0
    A = zeros((N,N))

    # A[0,0] = alpha/(alpha+1) - 1./(2*(alpha+1))
    # A[0,1] = 1./(2*(alpha+1))

    # A[N-1,N-1] = alpha/(alpha+1) + 1./(2*(alpha+1))
    # A[N-1,N-2] = 1./(2*(alpha+1))

    A[0,0] = 2./3
    A[0,1] = 1./3
    A[0,2] = 0
    A[N-1,N-1] = 2./3 
    A[N-1,N-2] = 1./3
    A[N-1,N-3] = 0


    for i in range(1,N-1):
        A[i,i] =  alpha/(alpha+1)
        A[i,i-1] = 1./(2*(alpha+1)) 
        A[i,i+1] = 1./(2*(alpha+1)) 

    e_val, e_vec = linalg.eig(A)

    index = argsort(e_val)
    order_val = array([e_val[i] for i in index])
    order_vec = array([e_vec[i,:] for i in index])

    plt.figure()
    plt.plot(order_val)
    plt.xlabel('t')
    plt.ylabel('eigenvalues')
    plt.title('Numerical eigenvalues')

    plt.figure()
    v = zeros(N)
    for i in range(0,N):
        v = v + order_vec[i,:]*abs(order_val[i]) 
    plt.plot(v)
    plt.xlabel('t')
    plt.ylabel('eigenvectors')
    plt.title('Numerical eigenvectors')
    plt.show()

