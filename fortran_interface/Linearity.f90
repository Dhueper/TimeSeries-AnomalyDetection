module Linearity
   
implicit none
    
    abstract interface  

       real function F_scalar(v)
            real, intent(in) :: v(:)
       end function
       
       real function F_1D(x, u, ux, uxx) 
            real, intent(in) :: x, u, ux, uxx                 
       end function 
       
       real function F_2D(x, y, u, ux, uy, uxx, uyy, uxy)
            real, intent(in) :: x, y, u, ux, uy, uxx, uyy, uxy
       end function
       
        function F_2D_System(x, y, u, ux, uy, uxx, uyy, uxy)
            real, intent(in) :: x, y, u(:), ux(:), uy(:), &
                                uxx(:), uyy(:), uxy(:)
            real :: F_2D_System(size(u)) 
        end function
       
       function F_vectorial(M,u)
            integer, intent(in) :: M
            real, intent(in) :: u(:)
            real :: F_vectorial(M)
       end function
            
    end interface

    contains

    
   
logical function Function_Linearity( F, N, P_u) result(L)

    procedure (F_scalar) :: F    
    integer, intent(in) :: N, P_u
    
    integer :: i
    real :: u(N), v(N), e1, e2
    real, parameter :: eps = 1e-6
    
    call random_number(u)
    call random_number(v)
    
    u(1:P_u-1) = 0
    v(1:P_u-1) = 0
    
    
!** First requirement e1 = 0
    e1 = F(u + v) - ( F(u) + F(v) )
    
!** Second requirement e2 = 0
    e2 = F(3 * u) - 3 * F(u)

    if (abs(e1) < eps .AND. abs(e2) < eps) then 
        L = .true.
    else
        L = .false.
    end if

end function
   
logical function Linearity_BVP_1D( F ) result(L)
    procedure (F_1D) :: F
    
    integer :: N = 4
    integer :: P_u = 2
    
    L = Function_Linearity( F_v, N, P_u)
    
   
contains

    real function F_v( u )
    
        real, intent(in) :: u(:)
        
        F_v = F(u(1), u(2), u(3), u(4))
        
    end function
    
end function

logical function Linearity_BVP_2D( F ) result(L)
    procedure (F_2D) :: F

    
    integer :: N = 8
    integer :: P_ux = 4
    
    
    L = Function_Linearity( F_v, N, P_ux)
    

contains

    real function F_v(u)
    
        real, intent(in) :: u(:)
        
        F_v = F ( u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8) )
    
    end function
    
end function

    
logical function Linearity_BVP_2D_System( M, F) result(L)
    integer, intent(in) :: M 
    procedure (F_2D_System) :: F
    
    logical :: L_aux(M)
    integer :: N, P_ux, i

    N = 2+6*M
    P_ux = 2 + M + 1
    
!** Obtain the linearity of each equation
    do i=1, M
        L_aux(i) = Function_Linearity( G, N, P_ux)
    end do
    
    L = ALL(L_aux)
    

contains

    real function G(u)
        real, intent(in) :: u(:)
        
        real :: aux(M)

        aux = F_v(M, u)

        G = aux(i)
    end function

    function F_v( M, u)
        integer, intent(in) :: M
        real, intent(in) :: u(:)
        real :: F_v(M)
    
        F_v = F(u(1), u(2), u(3:3+M-1), u(3+M:3+2*M-1), u(3+2*M:3+3*M-1), u(3+3*M:3+4*M-1), u(3+4*M:3+5*M-1), u(3+5*M:3+6*M-1))
    
    end function

end function

end module
    