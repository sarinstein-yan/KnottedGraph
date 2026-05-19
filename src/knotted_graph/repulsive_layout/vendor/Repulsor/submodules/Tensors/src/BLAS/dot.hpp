#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<typename Scal, IntQ I0, IntQ I1, IntQ I2>
        [[nodiscard]] TOOLS_FORCE_INLINE Scal dot(
            const I0 n_, cptr<Scal> x_, const I1 inc_x_,
                         cptr<Scal> y_, const I2 inc_y_
        )
        {
            Int n      = int_cast<Int>(n_);
            Int inc_x  = int_cast<Int>(inc_x_);
            Int inc_y  = int_cast<Int>(inc_y_);
            
            auto * x = to_BLAS(x_);
            auto * y = to_BLAS(y_);
            
            assert_positive(n);
            assert_positive(inc_x);
            assert_positive(inc_y);

            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_ddot( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sdot( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
#if defined(ACCELERATE_NEW_LAPACK)
                Scal result {0};
                
                cblas_zdotu_sub( n, x, inc_x, y, inc_y, &result );
                
                return result;
#else
                return cblas_zdot( n, x, inc_x, y, inc_y );
#endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_cdot( n, x, inc_x, y, inc_y );
                
#if defined(ACCELERATE_NEW_LAPACK)
                Scal result {0};
                
                cblas_cdotu_sub( n, x, inc_x, y, inc_y, &result );
                
                return result;
#else
                return cblas_cdot( n, x, inc_x, y, inc_y );
#endif
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors


