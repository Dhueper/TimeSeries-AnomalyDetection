module time_series
use Non_Linear_Systems 

implicit none
real(8), allocatable :: mvf_optim(:)
integer :: x_size
contains

! Mean value filter
! Time series filter based on the mean value theorem and a discrete integration rule:
! alpha=1 (linear), alpha=2 (quadratic).
! Input: x (fortran array), time series
!        alpha (real), order of the filter
! Returns: mvf (fortran array), filtered time series
function mvf(x, alpha) 
    implicit none
    real(8), intent(in):: x(:)
    real(8), INTENT(IN) :: alpha
    real(8) :: mvf(size(x))
    INTEGER :: j
    real(8) :: a1, a0

    a1 = 1d0/(2*(alpha+1))
    a0 = alpha/(alpha+1)

    mvf(1) = x(1)
    mvf(size(x)) = x(size(x))


    do j = 2,size(x)-1
        mvf(j) = a1*x(j-1) + a0*x(j) + a1*x(j+1)
    end do

end function

! Iterative mean value filter with linear BC
! Iterative time series filter based on the mean value theorem and a discrete integration rule:
! alpha=1 (linear), alpha=2 (quadratic).
! Input: x (fortran array), time series
!        alpha (real), order of the filter
!        n_iter (integer), number of iterations
! Returns: mvf_linearbc (fortran array), filtered time series with linear BC
function mvf_linearbc(x, alpha, n_iter) 
    implicit none
    real(8), intent(in):: x(:)
    real(8), INTENT(IN) :: alpha
    integer, INTENT(IN) :: n_iter
    real(8) :: mvf_linearbc(size(x))
    INTEGER :: i, j
    real(8) :: a1, a0

    a1 = 1d0/(2*(alpha+1))
    a0 = alpha/(alpha+1)

    mvf_linearbc(:) = x(:)

    do i = 1, n_iter
        do j = 2,size(x)-1
            mvf_linearbc(j) = a1*mvf_linearbc(j-1) + a0*mvf_linearbc(j) + a1*mvf_linearbc(j+1)
        end do
        mvf_linearbc(1) = 2*mvf_linearbc(2) - mvf_linearbc(3)
        mvf_linearbc(size(x)) = 2*mvf_linearbc(size(x)-1) - mvf_linearbc(size(x)-2) 
    end do
end function


! Iterative mean value filter with optimized BC
! Iterative time series filter based on the mean value theorem and a discrete integration rule:
! alpha=1 (linear), alpha=2 (quadratic).
! Input: x (fortran array), time series
!        alpha (real), order of the filter
!        n_iter (integer), number of iterations
! Returns: mvf_optimbc (fortran array), filtered time series with optimized BC
function mvf_optimbc(x, alpha, n_iter) 
    implicit none
    real(8), intent(in):: x(:)
    real(8), INTENT(IN) :: alpha
    integer, INTENT(IN) :: n_iter
    real(8) :: mvf_optimbc(size(x)), x0(4)
    INTEGER :: i, j
    real(8) :: a1, a0

    x_size = size(x)
    allocate(mvf_optim(x_size))

    a1 = 1d0/(2*(alpha+1))
    a0 = alpha/(alpha+1)

    mvf_optimbc(:) = x(:)

    do i = 1, n_iter
        do j = 2,size(x)-1
            mvf_optimbc(j) = a1*mvf_optimbc(j-1) + a0*mvf_optimbc(j) + a1*mvf_optimbc(j+1)
        end do
        x0 = [mvf_optimbc(1), mvf_optimbc(size(x)), mvf_optimbc(2), mvf_optimbc(size(x)-1)] 
        mvf_optim(:) = mvf_optimbc(:)
        call Newton(F, x0)
        mvf_optimbc(1) = x0(1)
        mvf_optimbc(size(x)) = x0(2)
        mvf_optimbc(2) = (x0(3) + mvf_optimbc(2)) / 2d0
        mvf_optimbc(size(x)-1) = (x0(4) + mvf_optimbc(size(x)-1)) / 2d0
    end do

    DEALLOCATE(mvf_optim)

end function

function F(xv)
    implicit none
    real(8), intent(in) :: xv(:)
    real(8) :: F(size(xv))
    
    real(8) :: x1, x2, x3, x4, var, delta_var, delta_d2x
    real(8) :: x_aux(x_size-1), y_aux(x_size-7)

    x1 = xv(1)
    x2 = xv(2)
    x3 = xv(3)
    x4 = xv(4)

    !End points
    x_aux(1) = x1 
    x_aux(2:x_size-1) = mvf_optim(2:x_size-1)
    var = variance(x_aux(2:x_size-1))
    delta_var = var - variance(x_aux)
    delta_d2x = (x_aux(2) - 2*x_aux(3) + x_aux(4)) - (x_aux(1) - 2*x_aux(2) + x_aux(3))
    F(1) = abs(delta_var) + abs(delta_d2x)

    x_aux(1) = x2
    delta_var = var - variance(x_aux)
    delta_d2x = (x_aux(x_size-1) - 2*x_aux(x_size-2) + x_aux(x_size-3)) - (x_aux(1) - 2*x_aux(x_size-1) + x_aux(x_size-2))
    F(2) = abs(delta_var) + abs(delta_d2x)

    !Near end points
    y_aux(1) = x3
    y_aux(2:x_size-7) = mvf_optim(5:x_size-4)
    var = variance(y_aux(2:x_size-7))
    delta_var = var - variance(y_aux)
    delta_d2x = (y_aux(2) - 2*y_aux(3) + y_aux(4)) - (y_aux(1) - 2*y_aux(2) + y_aux(3))
    F(3) = abs(delta_var) + abs(delta_d2x)

    y_aux(1) = x4
    delta_var = var - variance(y_aux)
    delta_d2x = (y_aux(x_size-7) - 2*y_aux(x_size-8) + y_aux(x_size-9)) - (y_aux(1) - 2*y_aux(x_size-7) + y_aux(x_size-8))
    F(4) = abs(delta_var) + abs(delta_d2x)

    contains


    function variance(y)
        implicit none
        real(8), intent(in) :: y(:)
        real(8) :: y2(size(y)), mu, variance
        
        mu = SUM(y) / size(y)
        y2 = y**2d0
        variance = (sum(y2) - size(y)*mu**2d0) / (size(y)-1)

    end function
        
end function


end module