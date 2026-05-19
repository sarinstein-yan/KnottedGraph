#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Op opA, typename Scal,
            IntQ I0, IntQ I1, IntQ I2, IntQ I3, IntQ I4
        >
        TOOLS_FORCE_INLINE void gemv(
            const I0 m_, const I1 n_,
            cref<Scal> alpha, cptr<Scal> A_, const I2 ldA_,
                              cptr<Scal> x_, const I3 incx_,
            cref<Scal> beta,  mptr<Scal> y_, const I4 incy_
        )
        {
            Int m    = int_cast<Int>(m_);
            Int n    = int_cast<Int>(n_);
            Int ldA  = int_cast<Int>(ldA_);
            Int incx = int_cast<Int>(incx_);
            Int incy = int_cast<Int>(incy_);
            
            auto * A = to_BLAS(A_);
            auto * x = to_BLAS(x_);
            auto * y = to_BLAS(y_);
            
            assert_positive(m);
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(incx);
            assert_positive(incy);
            
            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, alpha, A, ldA, x, incx, beta, y, incy );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, alpha, A, ldA, x, incx, beta, y, incy );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_zgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, to_BLAS(&alpha), A, ldA, x, incx, &beta, y, incy );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_cgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, to_BLAS(&alpha), A, ldA, x, incx, &beta, y, incy );
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors

