#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Side side, UpLo uplo, Op opA, Diag diag, typename Scal,
        IntQ I0, IntQ I1, IntQ I2, IntQ I3
        >
        TOOLS_FORCE_INLINE void trsm(
            const I0 n_, const I1 nrhs_,
            cref<Scal> alpha, cptr<Scal> A_, const I2 ldA_,
                              mptr<Scal> B_, const I3 ldB_
        )
        {
            Int n    = int_cast<Int>(n_);
            Int nrhs = int_cast<Int>(nrhs_);
            Int ldA  = int_cast<Int>(ldA_);
            Int ldB  = int_cast<Int>(ldB_);
            
            auto * A = to_BLAS(const_cast<Scal*>(A_));
            auto * B = to_BLAS(B_);
            
            assert_positive(n);
            assert_positive(nrhs);
            assert_positive(ldA);
            assert_positive(ldB);
            
            
            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dtrsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, alpha, A, ldA, B, ldB );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_strsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, alpha, A, ldA, B, ldB );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_ztrsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, to_BLAS(&alpha), A, ldA, B, ldB );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_ctrsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, to_BLAS(&alpha), A, ldA, B, ldB );
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
        }
        
    } // namespace BLAS

} // namespace Tensors

