#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Op opA, Op opB,
            typename Scal, IntQ I0, IntQ I1, IntQ I2, IntQ I3, IntQ I4, IntQ I5
        >
        TOOLS_FORCE_INLINE void gemm(
            const I0 m_, const I1 n_, const I2 k_,
            cref<Scal> alpha, cptr<Scal> A_, const I3 ldA_,
                              cptr<Scal> B_, const I4 ldB_,
            cref<Scal> beta,  mptr<Scal> C_, const I5 ldC_
        )
        {
            Int m    = int_cast<Int>(m_);
            Int n    = int_cast<Int>(n_);
            Int k    = int_cast<Int>(k_);
            Int ldA  = int_cast<Int>(ldA_);
            Int ldB  = int_cast<Int>(ldB_);
            Int ldC  = int_cast<Int>(ldC_);
            
            auto * A = to_BLAS(A_);
            auto * B = to_BLAS(B_);
            auto * C = to_BLAS(C_);
            
//                logprint( std::string("BLASS::gemm ( ") + ToString(m) + "," + ToString(n) + "," + ToString(k) + " )");
                
            assert_positive(m);
            assert_positive(n);
            assert_positive(k);
            assert_positive(ldA);
            assert_positive(ldB);
            assert_positive(ldC);

            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_zgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, to_BLAS(&alpha), A, ldA, B, ldB, to_BLAS(&beta), C, ldC
                );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_cgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, to_BLAS(&alpha), A, ldA, B, ldB, to_BLAS(&beta), C, ldC
                );
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
        }
        
    } // namespace BLAS
    
} // namespace Tensors
