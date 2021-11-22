module Dependencies
   
implicit none
    
    abstract interface  

       real function F_scalar(u)
            real, intent(in) :: u(:)
       end function
       
       real function F_1D_BVP(x, u, ux, uxx) 
            real, intent(in) :: x, u, ux, uxx                 
       end function 
       
       real function F_2D(x, y, u, ux, uy, uxx, uyy, uxy)
            real, intent(in) :: x, y, u, ux, uy, uxx, uyy, uxy
       end function
       
        function F_2D_System(x, y, u, ux, uy, uxx, uyy, uxy)
            real, intent(in) :: x, y, u(:), ux(:), uy(:), uxx(:), uyy(:), uxy(:)
            real :: F_2D_System(size(u)) 
        end function
       
       function F_vectorial(M,u)
            integer, intent(in) :: M
            real, intent(in) :: u(:)
            real :: F_vectorial(M)
       end function
       real function F_1D_IBVP(x, t, u, ux, uxx) 
            real, intent(in) :: x, t, u, ux, uxx                 
       end function 
        real function F_2D_IBVP(x, y, t, u, ux, uy, uxx, uyy, uxy) 
            real, intent(in) :: x, y, t, u, ux, uy, uxx, uyy, uxy
       end function  

    end interface
    
    contains
   
    
    
    

    
    
function Function_Dependencies( F, N, P_ux ) result(D)

    procedure (F_scalar) :: F
    integer, intent(in) :: N, P_ux 
    logical :: D(1:N-P_ux+1) 
    
    integer :: i, j
    real :: h( N ), dF, u( N ), dx, r(2)
    logical :: Daux( 1:N, 1:2)
    
    
    call random_number(r)
    dx = 0.04
    
    do i = P_ux, n
        
        do j = 1, 2
            u = r(j)
            h = 0
            h(i) = 1
            dF=F(u + h * dx) - F(u)
                if(dF /= 0) then
                    Daux(i,j) = .true.
                else
                    Daux(i,j) = .false.
                end if
        end do

        if(Daux(i,1) .eqv. Daux(i,2)) then
            D(i-P_ux+1) = Daux(i,1)
        else 
            D(i-P_ux+1) = .true.
        end if
    end do

end function









!*****     
function Dependencies_BVP_1D( F ) result(d)
    procedure (F_1D_BVP) :: F
    logical :: d( 2 )
    
    integer :: N = 4
    integer :: P_ux = 3
    
    d = Function_Dependencies( F_v, N, P_ux)
        
contains

    real function F_v( u )
    
        real, intent(in) :: u(:)
        
        F_v = F(u(1), u(2), u(3), u(4))
        
    end function
    
end function
    
function Dependencies_BVP_2D( F ) result(D)
    procedure (F_2D) :: F
    logical :: D(5)
    
    integer :: N = 8
    integer :: P_ux = 4
    
    
    D = Function_Dependencies( F_v, N, P_ux)

contains

    real function F_v(u)
    
        real, intent(in) :: u(:)
        
        F_v = F ( u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8) )
    
    end function
    
end function







!*****     
function Dependencies_BVP_2D_System( M, F) result(d)
    integer, intent(in) :: M 
    procedure (F_2D_System) :: F
    logical :: d(M,5)
    
    logical :: d_aux1(M, 5*M), d_aux2(5*M)
    integer :: N, P_ux, i, j, k

    N = 2+6*M
    P_ux = 3+M 
    
    do i=1, M
        d_aux1(i,:) = Function_Dependencies( G, N, P_ux)
    end do


    do i=1, 5*M
        d_aux2(i) = any( d_aux1(:,i) )
    end do
    

    do i=1, M
        do j=0, 4
            k=i+j*M
            d(i, j+1) = d_aux2(k)
        end do
    end do
    
        
    

contains

    function F_v( M, u)
        integer, intent(in) :: M
        real, intent(in) :: u(:)
        real :: F_v(M)
    
        F_v = F(u(1), u(2), u(3:3+M-1), u(3+M:3+2*M-1), u(3+2*M:3+3*M-1), u(3+3*M:3+4*M-1), u(3+4*M:3+5*M-1), u(3+5*M:3+6*M-1))
    
    end function

    real function G(u)
        real, intent(in) :: u(:)
        
        real :: aux(M)

        aux = F_v(M, u)

        G = aux(i)
    end function

end function

 






!*****     
function Dependencies_IBVP_1D( F ) result(d)
    procedure (F_1D_IBVP) :: F
    logical :: d( 2 )
    
    integer :: N = 5
    integer :: P_ux = 4
    
    d = Function_Dependencies( F_v, N, P_ux)
        
contains

    real function F_v( u )
    
        real, intent(in) :: u(:)
        
        F_v = F(u(1), u(2), u(3), u(4), u(5))
        
    end function
end function


function Dependencies_IBVP_2D( F ) result(D)
    procedure (F_2D_IBVP) :: F
    logical :: D(5)
    
    integer :: N = 9
    integer :: P_ux = 5
    
    
    D = Function_Dependencies( F_v, N, P_ux)

contains

    real function F_v(u)
    
        real, intent(in) :: u(:)
        
        F_v = F ( u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8), u(9) )
    
    end function
    
end function

end module