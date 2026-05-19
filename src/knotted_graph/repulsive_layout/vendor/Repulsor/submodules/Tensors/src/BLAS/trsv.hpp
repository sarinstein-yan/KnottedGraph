#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, UpLo uplo, Op opA, Diag diag, typename Scal,
            IntQ I0, IntQ I1, IntQ I2
        >
        TOOLS_FORCE_INLINE void trsv(
            const I0 n_, cptr<Scal> A_, const I1 ldA_,
                         mptr<Scal> x_, const I2 incx_
        )
        {
            Int n    = int_cast<Int>(n_);
            Int ldA  = int_cast<Int>(ldA_);
            Int incx = int_cast<Int>(incx_);
            
            auto * A = to_BLAS(const_cast<Scal*>(A_));
            auto * x = to_BLAS(x_);
            
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(incx);
            
            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dtrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, A, ldA, x, incx );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_strsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, A, ldA, x, incx );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_ztrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, A, ldA, x, incx );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_ctrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, A, ldA, x, incx );
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
        
} // namespace Tensors


