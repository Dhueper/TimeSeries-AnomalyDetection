module time_series_py
use time_series
implicit none
contains
subroutine mvf_py(x,alpha,mvf_pye,N_x)
real(8),dimension(N_x), intent(in) :: x
real(8), INTENT(IN)  :: alpha
integer, intent(in) :: N_x
real(8),dimension(size(x)),intent(out) :: mvf_pye
mvf_pye = mvf(x,alpha)
end subroutine
subroutine mvf_linearbc_py(x,alpha,n_iter,mvf_linearbc_pyj,N_x)
real(8),dimension(N_x), intent(in) :: x
real(8), INTENT(IN)  :: alpha
integer, INTENT(IN)  :: n_iter
integer, intent(in) :: N_x
real(8),dimension(size(x)),intent(out) :: mvf_linearbc_pyj
mvf_linearbc_pyj = mvf_linearbc(x,alpha,n_iter)
end subroutine
subroutine mvf_optimbc_py(x,alpha,n_iter,mvf_optimbc_pyb,N_x)
real(8),dimension(N_x), intent(in) :: x
real(8), INTENT(IN)  :: alpha
integer, INTENT(IN)  :: n_iter
integer, intent(in) :: N_x
real(8),dimension(size(x)),intent(out) :: mvf_optimbc_pyb
mvf_optimbc_pyb = mvf_optimbc(x,alpha,n_iter)
end subroutine
subroutine F_py(xv,F_pyj,N_xv)
real(8),dimension(N_xv), intent(in)  :: xv
integer, intent(in) :: N_xv
real(8),dimension(size(xv)),intent(out) :: F_pyj
F_pyj = F(xv)
end subroutine
end module