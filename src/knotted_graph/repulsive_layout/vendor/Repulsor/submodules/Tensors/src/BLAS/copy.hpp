#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<typename Scal, IntQ I0, IntQ I1, IntQ I2>
        TOOLS_FORCE_INLINE void copy(
            const I0 n_, cptr<Scal> x_, const I1 inc_x_,
                         mptr<Scal> y_, const I2 inc_y_
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
                return cblas_dcopy( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_scopy( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_ccopy( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_zcopy( n, x, inc_x, y, inc_y );
            }
            else
            {
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors



