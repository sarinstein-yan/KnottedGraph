#pragma once

namespace Tensors
{
    namespace BLAS
    {
        
        template<
            Layout layout, UpLo uplo, Op op, typename Scal,
            IntQ I0, IntQ I1, IntQ I2, IntQ I3
        >
        TOOLS_FORCE_INLINE void herk(
            const I0 n_, const I1 k_,
            cref<Scalar::Real<Scal>> alpha, cptr<Scal> A_, const I2 ldA_,
            cref<Scalar::Real<Scal>> beta , mptr<Scal> C_, const I3 ldC_
        )
        {
            Int n    = int_cast<Int>(n_);
            Int k    = int_cast<Int>(k_);
            Int ldA  = int_cast<Int>(ldA_);
            Int ldC  = int_cast<Int>(ldC_);
            
            auto * A = to_BLAS(A_);
            auto * C = to_BLAS(C_);
                        
            assert_positive(n);
            assert_positive(k);
            assert_positive(ldA);
            assert_positive(ldC);
            
            // If op == Op::Id, then
            // this computes the upper or lower triangle part (depending on uplo) of
            //
            //      C := alpha* A * A^H + beta * C,
            //
            // where A is a matrix of size n x k and where C is a self-adjoint matrix of size n x n
            // (of which we assume only that the uplo part is defined).

            // If op == Op::ConjTrans, then
            // this computes the upper or lower triangle part of
            //
            //      C := alpha* A^H * A + beta * C,
            //
            // where A is now a matrix of size k x n and C is again a self-adjoint matrix of size n x n.
            
            // We also overload this routine to work with real scalars (float and double)
            // and redirect the work to ssyrk and dsyrk, respectively.
            
            // Moreover, we graciously interpret Op::Trans as Op::ConjTrans.
            
            if constexpr ( SameQ<Scal,double> )
            {
                if constexpr ( op == Op::Id )
                {
                    return cblas_dsyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_dsyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Trans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                if( op == Op::Id )
                {
                    return cblas_ssyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_ssyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Trans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                if( op == Op::Id )
                {
                    return cblas_zherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_zherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::ConjTrans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                if( op == Op::Id )
                {
                    return cblas_cherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_cherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::ConjTrans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors

