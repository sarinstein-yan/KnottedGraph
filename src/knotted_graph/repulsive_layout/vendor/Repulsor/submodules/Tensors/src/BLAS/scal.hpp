#pragma once

namespace Tensors
{

    namespace BLAS
    {
        template<typename Scal, IntQ I0, IntQ I1>
        TOOLS_FORCE_INLINE void scal(
            const I0 n_, cref<Scal> alpha, mptr<Scal> x_, const I1 inc_x_
        )
        {
            Int n      = int_cast<Int>(n_);
            Int inc_x  = int_cast<Int>(inc_x_);
            
            auto * x = to_BLAS(x_);
            
            assert_positive(n);
            assert_positive(inc_x);

            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dscal( n, alpha, x, inc_x );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_dscal( n, alpha, x, inc_x );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_cscal( n, to_BLAS(&alpha), x, inc_x );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_zscal( n, to_BLAS(&alpha), x, inc_x );
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors


