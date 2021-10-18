from numpy import linalg, linspace, pi, cos, sin, zeros, sort, argsort, array, matmul, dot
from matplotlib import pyplot as plt

import test_function

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
    for alpha in range(0,3):
        analytical_ev(alpha)
        l.append('alpha='+str(alpha))
    analytical_ev(0.5)
    l.append('moving averages')
    plt.legend(l)
    plt.xlabel('t')
    plt.ylabel('eigenvalues')
    plt.title('Analytical eigenvalues')

    #Numerical
    N = 500

    t = linspace(0,1,N)
    # X = t**3. + cos(2*pi*50 * t)
    X = 1 + t*0
    # X = sin(2*pi*10 * t)

    alpha = 1
    A = zeros((N,N))

    A[0,0] = alpha/(alpha+1) - 1./(2*(alpha+1))
    A[0,1] = 1./(2*(alpha+1))

    A[N-1,N-1] = alpha/(alpha+1) - 1./(2*(alpha+1))
    A[N-1,N-2] = 1./(2*(alpha+1))

    # A[0,0] = alpha/(alpha+1)
    # A[0,1] = 1./(2*(alpha+1))
    # A[0,2] = 0
    # A[N-1,N-1] = alpha/(alpha+1)
    # A[N-1,N-2] = 1./(2*(alpha+1))
    # A[N-1,N-3] = 0


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
    sum_v = zeros(N)
    for i in range(0,int(N)):
        v[i] = dot(X,order_vec[i,:])
        sum_v = sum_v + v[i]* order_vec[i,:]*order_val[i]**50. 
    plt.plot(sum_v)
    plt.xlabel('t')
    plt.ylabel('eigenvectors')
    plt.title('Numerical eigenvectors')

    #Test 
    plt.figure()
    plt.plot(t,X)
    Y = matmul(linalg.matrix_power(A,50),X)
    # Y2 = zeros(N)
    # for i in range(0,4):
    #     Y2[0] =  (2*alpha*X[0] + X[1])/(2* (alpha+1))  
    #     Y2[N-1] =  (2*alpha*X[N-1] + X[N-2])/(2* (alpha+1)) 
    #     for j in range(1,N-1):
    #         Y2[j] = (X[j-1] + 2*alpha*X[j] + X[j+1])/(2 * (alpha+1))  

    plt.plot(t,Y)
    # plt.plot(t,Y2)
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.title('Test function')
    plt.show()


