#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Op opx, Op opy, typename Scal,
            IntQ I0, IntQ I1, IntQ I2, IntQ I3, IntQ I4
        >
        TOOLS_FORCE_INLINE void ger(
            const I0 m_, const I1 n_,
            cref<Scal> alpha, cptr<Scal> x_, const I2 inc_x_,
                              cptr<Scal> y_, const I3 inc_y_,
                              mptr<Scal> A_, const I4 ldA_
        )
        {
            // A := alpha * opx(x) * opx(y^T) + A,
            //
            // where {opx,opy} can be {id,id}, {conj,id}, or {id, conj}.
            // Note that {conj, conj} is _not_ allowed (cblas does not allow us to contruct it.

            Int m     = int_cast<Int>(m_);
            Int n     = int_cast<Int>(n_);
            Int ldA   = int_cast<Int>(ldA_);
            Int inc_x = int_cast<Int>(inc_x_);
            Int inc_y = int_cast<Int>(inc_y_);
            
            auto * A = to_BLAS(A_);
            auto * x = to_BLAS(x_);
            auto * y = to_BLAS(y_);
            
            assert_positive(m);
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(inc_x);
            assert_positive(inc_y);


            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dger( to_BLAS(layout), m, n, alpha, x, inc_x, y, inc_y, A, ldA );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sger( to_BLAS(layout), m, n, alpha, x, inc_x, y, inc_y, A, ldA );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                if constexpr ( (opx == Op::Id) && (opy == Op::Id) )
                {
                    return cblas_zgeru(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Conj) && (opy == Op::Id) )
                {
                    return cblas_zgerc(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, to_BLAS(&alpha), x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Id) && (opy == Op::Conj) )
                {
                    return cblas_zgerc(
                        to_BLAS(layout),
                        m, n, to_BLAS(&alpha), x, inc_x, y, inc_y, A, ldA
                    );
                }
                else
                {
                    static_assert(Tools::DependentFalse<Scal>,"ger does not allow us to conjugate both input vectors.");
                }
    
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                if constexpr ( (opx == Op::Id) && (opy == Op::Id) )
                {
                    return cblas_cgeru(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, to_BLAS(&alpha), x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Conj) && (opy == Op::Id) )
                {
                    return cblas_cgerc(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, to_BLAS(&alpha), x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Id) && (opy == Op::Conj) )
                {
                    return cblas_cgerc(
                        to_BLAS(layout),
                        m, n, to_BLAS(&alpha), x, inc_x, y, inc_y, A, ldA
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

