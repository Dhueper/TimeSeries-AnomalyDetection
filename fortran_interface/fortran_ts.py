import fortran_tsf
class time_series:
    def mvf(x,alpha):
        """
            ! Mean value filter
            ! Time series filter based on the mean value theorem and a discrete integration rule:
            ! alpha=1 (linear), alpha=2 (quadratic).
            ! Input: x (fortran array), time series
            !        alpha (real), order of the filter
            ! Returns: mvf (fortran array), filtered time series
            x (real(8)): None
            alpha (real(8)): None
        """
        return fortran_tsf.time_series_py.mvf_py(x,alpha)
    def mvf_linearbc(x,alpha,n_iter):
        """
            ! Iterative mean value filter with linear BC
            ! Iterative time series filter based on the mean value theorem and a discrete integration rule:
            ! alpha=1 (linear), alpha=2 (quadratic).
            ! Input: x (fortran array), time series
            !        alpha (real), order of the filter
            !        n_iter (integer), number of iterations
            ! Returns: mvf_linearbc (fortran array), filtered time series with linear BC
            x (real(8)): None
            alpha (real(8)): None
            n_iter (integer): None
        """
        return fortran_tsf.time_series_py.mvf_linearbc_py(x,alpha,n_iter)
    def mvf_optimbc(x,alpha,n_iter):
        """
            ! Iterative mean value filter with optimized BC
            ! Iterative time series filter based on the mean value theorem and a discrete integration rule:
            ! alpha=1 (linear), alpha=2 (quadratic).
            ! Input: x (fortran array), time series
            !        alpha (real), order of the filter
            !        n_iter (integer), number of iterations
            ! Returns: mvf_optimbc (fortran array), filtered time series with optimized BC
            x (real(8)): None
            alpha (real(8)): None
            n_iter (integer): None
        """
        return fortran_tsf.time_series_py.mvf_optimbc_py(x,alpha,n_iter)
    def F(xv):
        """
            xv (real(8)): None
        """
        return fortran_tsf.time_series_py.f_py(xv)