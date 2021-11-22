module time_series_py
use time_series
implicit none
contains
subroutine mvf_py(x,alpha,mvf_pyu,N_x)
real(8),dimension(N_x), intent(in) :: x
real(8), INTENT(IN)  :: alpha
integer, intent(in) :: N_x
real(8),dimension(size(x)),intent(out) :: mvf_pyu
mvf_pyu = mvf(x,alpha)
end subroutine
subroutine mvf_linearbc_py(x,alpha,n_iter,mvf_linearbc_pyd,N_x)
real(8),dimension(N_x), intent(in) :: x
real(8), INTENT(IN)  :: alpha
integer, INTENT(IN)  :: n_iter
integer, intent(in) :: N_x
real(8),dimension(size(x)),intent(out) :: mvf_linearbc_pyd
mvf_linearbc_pyd = mvf_linearbc(x,alpha,n_iter)
end subroutine
end module