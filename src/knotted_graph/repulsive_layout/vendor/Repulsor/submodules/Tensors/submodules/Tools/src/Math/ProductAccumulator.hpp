#pragma once

namespace Tools
{
    template<typename Scal_, IntQ Int_ = Int64>
    class ProductAccumulator final
    {
        using Scal    = Scal_;
        using Int     = Int_;
        using Real    = Scalar::Real<Scal>;
        using Complex = Scalar::Complex<Real>;
        
        static_assert( Scalar::RealQ<Scal> || Scalar::ComplexQ<Scal>, "" );
        
    private:
        
        Real x = 0.5; // mantissa of real part
        Real y = 0;   // mantissa of imag part
        
//        Int e_x = 1; // mantissa of real part
//        Int e_y = 0; // mantissa of imag part
        
        Int  e = 1; // common exponent
        
    public:
        
        ProductAccumulator() = default;
        
//        ProductAccumulator( const Scal init, Int shift )
//        {
//            if constexpr ( Scalar::RealQ<Scal> )
//            {
//                std::tie(x,e)   = mantissa_exponent( init );
//            }
//            else
//            {
//                std::tie(x,y,e) = mantissa_exponent( init );
//            }
//            
//            e += shift;
//        }
        
        ProductAccumulator( cref<Scal> init )
        {
            if constexpr ( Scalar::RealQ<Scal> )
            {
                std::tie(x,e)   = mantissa_exponent( init );
            }
            else
            {
                std::tie(x,y,e) = mantissa_exponent( init );
            }
        }
        
        ProductAccumulator( cref<Scal>  mantissa_init, cref<Int> exponent_init )
        :   ProductAccumulator( mantissa_init )
        {
            e += exponent_init;
        }

        ~ProductAccumulator() = default;
        
        
    private:
        
        std::tuple<Real,Int> mantissa_exponent( const Real z )
        {
            int exponent;
            Real mantissa = std::frexp( z, &exponent );
            
            return std::tuple<Real,Int>(mantissa,exponent);
        }
        
        std::tuple<Real,Real,Int> mantissa_exponent( const Complex z )
        {
            auto [u,e_u] = mantissa_exponent( Re(z) );
            auto [v,e_v] = mantissa_exponent( Im(z) );
            
            const Int d = e_u - e_v;
            
            Int exponent;
            
            const bool use_u_e = ((d >= Int(0)) && (u != Int(0))) || (v == Int(0));
            
            if( use_u_e )
            {
                exponent = e_u;
                v = v * Tools::Power(Real(2),-d);
            }
            else
            {
                exponent = e_v;
                u = u * Tools::Power(Real(2),d);
            }
            
            return std::tuple<Real,Real,Int>(u,v,exponent);
        }
        
    public:
        
        ProductAccumulator & operator*=( const Scal z )
        {
            Int shift;
            
            if constexpr ( Scalar::RealQ<Scal> )
            {
                std::tie(x,shift) = mantissa_exponent( x * Re(z) );
                e += shift;
            }
            else
            {
                std::tie(x,y,shift) = mantissa_exponent( Complex(x,y) * z );
                e += shift;
            }
            
            return *this;
        }
        
        ProductAccumulator & operator*=( const ProductAccumulator & z )
        {
            e += z.e;
            
            if constexpr ( Scalar::RealQ<Scal> )
            {
                return (*this) *= z.x;
            }
            else
            {
                return (*this) *= Complex(z.x,z.y);
            }
        }
        
        
        friend ProductAccumulator operator*(
            const ProductAccumulator & a, const ProductAccumulator & b
        )
        {
            ProductAccumulator c (a);
            
            c *= b;
            
            return c;
        }
        
        ProductAccumulator & operator/=( const Scal z )
        {
            Int shift = 0;
            
            if constexpr ( Scalar::RealQ<Scal> )
            {
                std::tie(x,shift) = mantissa_exponent( x / Re(z) );
                e += shift;
            }
            else
            {
                std::tie(x,y,shift) = mantissa_exponent( Complex(x,y) / z );
                e += shift;
            }
            
            return *this;
        }
        
        ProductAccumulator & operator/=( const ProductAccumulator & z )
        {
            e -= z.Exponent2();
            
            return (*this) /= z.Mantissa2();
        }

        
        friend ProductAccumulator operator/(
            const ProductAccumulator & a, const ProductAccumulator & b
        )
        {
            ProductAccumulator c (a);
            
            c /= b;
            
            return c;
        }
        
        friend ProductAccumulator<Real,Int> Abs( const ProductAccumulator<Scal,Int> z )
        {
            return Abs(z.Mantissa2()) * ProductAccumulator<Real,Int>::Power(Real(2),z.Exponent2());
        }
        
        ProductAccumulator Power( const Int exponent ) const
        {
            if constexpr ( Scalar::RealQ<Scal> )
            {
                Real log2_bx = std::log2(x) * exponent;
                
                Int q = std::trunc(log2_bx);

                ProductAccumulator<Real,Int> b;
                b.x = std::exp2(log2_bx - q);
                b.e = e * exponent + q;
                
                // Normalization step. Maybe obsolete.
                int delta;
                
                b.x  = std::frexp( b.x, &delta );
                b.e += delta;
                
                return b;
                
            }
            else
            {
                // magnitude of mantissa
                ProductAccumulator<Real,Int> r = Abs(*this);
                
                if( r.Mantissa2() == Real(0) )
                {
                    return ProductAccumulator<Complex,Int>(0);
                }
                
                // magnitude of mantissa^exponent
                ProductAccumulator<Real,Int> R = r.Power(exponent);
                
                // argument of mantissa^exponent
                Real Theta = exponent * std::atan2(y,x);
                
                ProductAccumulator<Complex,Int> b;
                
                b.x = R.Mantissa2() * std::cos( Theta );
                b.y = R.Mantissa2() * std::sin( Theta );
                b.e = R.Exponent2();
                
                return b;
            }
            
        }
        
        static ProductAccumulator Power( const Scal & base, Int e )
        {
            ProductAccumulator b (base);
            
            return b.Power(e);
        }
        
        Scal Mantissa2() const
        {
            if constexpr ( Scalar::RealQ<Scal> )
            {
                return x;
            }
            else
            {
                return Complex(x,y);
            }
        }
        
        Int Exponent2() const
        {
            return e;
        }
        
        std::pair<Scal,Int> MantissaExponent2()  const
        {
            if constexpr ( Scalar::RealQ<Scal> )
            {
                return std::pair<Real,Int>( x, e );
            }
            else
            {
                return std::pair<Complex,Int>( Complex(x,y), e );
            }
        }
        
        Scal Mantissa10() const
        {
            return MantissaExponent10().first;
        }
        
        Int Exponent10() const
        {
            return MantissaExponent10().second;
        }
        
        std::pair<Scal,Int> MantissaExponent10()  const
        {
            if( x == Scal(0) )
            {
                return std::pair(Scal(0),Int(0));
            }
            
//            constexpr Real conversion_factor = std::log(Real(2)) / std::log(Real(10));
            constexpr Real conversion_factor = 0.3010299956639812;
            
            auto exp10 = []( const Real x_ )
            {
                return std::pow( Real(10), x_ );
            };
            
            if constexpr ( Scalar::RealQ<Scal> )
            {
                Real exponent10 = e * conversion_factor;
                Int  e10 = std::trunc(exponent10);
                Real x10 = x * exp10( exponent10 - e10 );

                while( Abs(x10) < Real(1) )
                {
                    x10 *= Real(10);
                    ++e10;
                }
                
                while( Abs(x10) >= Real(10) )
                {
                    x10 *= Frac<Real>(1,10);
                    ++e10;
                }
                
                return std::pair<Real,Int>( x10, e10 );
            }
            else
            {
                Real    exponent10 = e * conversion_factor;
                Int     e10 = std::trunc(exponent10);
                Complex z10 = Complex(x,y) * exp10( exponent10 - e10 );
                
                while( AbsSquared(z10) < Real(1) )
                {
                    z10 *= Real(10);
                    --e10;
                }
                
                while( AbsSquared(z10) >= Real(100) )
                {
                    z10 *= Frac<Real>(1,10);
                    ++e10;
                }
                
                return std::pair<Complex,Int>( z10, e10 );
            }
        }


        [[nodiscard]] friend std::string ToString( cref<ProductAccumulator> a ) 
        {
            if constexpr ( Scalar::RealQ<Scal> )
            {
                return std::string("{ ") + ToString(a.x) + ", " + ToString(a.e) + " } ";
            }
            else
            {
                return std::string("{ ") + ToString(a.x) + ", " + ToString(a.y) + ", " + ToString(a.e) + " } ";
            }
        }
        
    }; // class ProduceAccumulator
    
} // namespace Tools
