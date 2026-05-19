#pragma once

namespace Tools
{
    template<typename Real, NonIntQ T>
    inline Real Power( const Real base, const T exponent )
    {
        // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
        return (base > Scalar::Zero<Real>) ? std::exp2( static_cast<Real>(exponent) * std::log2(base) ) : ( static_cast<Real>(exponent)!=Scalar::Zero<Real> ? Scalar::Zero<Real> : Scalar::One<Real> );
    } // Power
    
    template<typename Real, IntQ Int>
    inline Real Power( const Real base, const Int exponent)
    {
        if( exponent >= Int(0))
        {
            switch( exponent )
            {
                case 0:
                {
                    return Scalar::One<Real>;
                }
                case 1:
                {
                    return base;
                }
                case 2:
                {
                    return base * base;
                }
                case 3:
                {
                    return base * base * base;
                }
                case 4:
                {
                    Real b2 = base * base;
                    return b2 * b2;
                }
                case 5:
                {
                    Real b2 = base * base;
                    return b2 * b2 * base;
                }
                case 6:
                {
                    Real b2 = base * base;
                    return b2 * b2 * b2;
                }
                case 7:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b2 * base;
                }
                case 8:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4;
                }
                case 9:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * base;
                }
                case 10:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * b2;
                }
                case 11:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * b2 * base;
                }
                case 12:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * b4;
                }

                default:
                {
                    Real exp = exponent;
                    return Power(base, exp);
                }
            }
        }
        else
        {
            return Inv( Power(base, -exponent) );
        }
    } // Power
    
    template<typename R = FastInt8, typename T>
    TOOLS_FORCE_INLINE constexpr R Sign( const T x)
    {
        constexpr T zero = 0;
        
        return static_cast<R>(x > zero) - static_cast<R>(x < zero);
    }
    
    template<typename R = FastInt8, typename S, typename T>
    TOOLS_FORCE_INLINE constexpr R DifferenceSign( const S x, const T y )
    {
        // Returns the sign of x - y.
        
        return static_cast<R>(x > y) - static_cast<R>(x < y);
    }
    
    template<Size_T n, typename T>
    TOOLS_FORCE_INLINE constexpr
    std::enable_if_t<VectorizableQ<T>,decltype(vec_T<n,T>{} > vec_T<n,T>{})>
    DifferenceSign( cref<vec_T<n,T>> x, cref<vec_T<n,T>> y )
    {
        // Returns the sign of x - y.
        
        return (x > y) - (x < y);
    }
    
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr bool OppositeSignQ( const T x, const T y )
    {
        // This is often adviced, but it is wrong.
        // It returns true for ( -1 ^ 0 ) < 0.
//        return (x ^ y) < 0;
        
        return (x * y < 0);
    }
    
    
    template<typename R = FastInt8, typename S, typename T>
    TOOLS_FORCE_INLINE constexpr R KroneckerDelta( const S i, const T j )
    {
        return static_cast<R>( i == j );
    }
    
    
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr bool Equal3( const T & a, const T & b, const T & c )
    {
        return (a == b) && (b == c);
    }
    
    
    template<typename Scal>
    TOOLS_FORCE_INLINE constexpr Scal Mean( const Scal & x, const Scal & y )
    {
        using Real = Scalar::Real<Scal>;
     
        static_assert(FloatQ<Real>,"");
        
        return Scalar::Half<Real> * (x + y);
    }
    
    
    // Max
    
    // We use std::fmax instead of std::max.
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Max( const Real & x, const Real & y )
    {
        using std::max;
        
        if constexpr ( std::is_floating_point_v<Real> )
        {
            return std::fmax(x,y);
        }
        else
        {
            return max(x,y);
        }
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Max( const Real & x, const Real & y, const Real & z )
    {
        using std::max;
        
        if constexpr ( std::is_floating_point_v<Real> )
        {
            return std::fmax(std::fmax(x,y),z);
        }
        else
        {
            return max({x,y,z});
        }
    }
    

    // Min
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Min( const Real & x, const Real & y )
    {
        using std::min;
        
        if constexpr ( std::is_floating_point_v<Real> )
        {
            return std::fmin(x,y);
        }
        else
        {
            return min(x,y);
        }
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Min( const Real & x, const Real & y, const Real & z )
    {
        using std::min;
        
        if constexpr ( std::is_floating_point_v<Real> )
        {
            return std::fmin(std::fmin(x,y),z);
        }
        else
        {
            return min({x,y,z});
        }
    }
    
    // MinMax
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr std::pair<Real,Real> MinMax( const Real & x, const Real & y )
    {
        using std::minmax;
        
        if constexpr ( std::is_floating_point_v<Real> )
        {
            return std::pair<Real,Real>( std::fmin(x,y), std::fmax(x,y) );
        }
        else
        {
            return minmax(x,y);
        }
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr std::pair<Real,Real> MinMax( const std::pair<Real,Real> & p )
    {
        return MinMax( p.first, p.second );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr std::pair<Real,Real> MinMax(
        const Real & x, const Real & y, const Real & z
    )
    {
        using std::minmax;
        
        if constexpr ( std::is_floating_point_v<Real> )
        {
            return { std::fmin(std::fmin(x,y),z), std::fmax(std::fmax(x,y),z) };
        }
        else
        {
            return minmax({x,y,z});
        }
    }
    
    
    
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Ramp( const Real & x )
    {
        static_assert(Scalar::RealQ<Real>, "Argument must be of a real type.");
        return Max( Scalar::Zero<Real>, x );
    }

    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Clamp( const Real & x, const Real & a, const Real & b )
    {
        static_assert(Scalar::RealQ<Real>, "Argument must be of a real type.");
        return Max( a, Min( b, x ) );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Clip( const Real & x, const Real & a, const Real & b )
    {
        static_assert(Scalar::RealQ<Real>, "Argument must be of a real type.");
        return Max( a, Min( b, x ) );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Ramp_1( const Real & x )
    {
        static_assert(Scalar::RealQ<Real>, "Argument must be of a real type.");
        return Max( Scalar::One<Real>, x );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real Sqrt( const Real & x )
    {
        return std::sqrt(x);
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE constexpr Real InvSqrt( const Real & x )
    {
        return Inv( Sqrt(x) );
    }
    
    template<int AmbDim, typename Real>
    inline Real AngleBetweenUnitVectors( cptr<Real> x, cptr<Real> y )
    {
        Real a = Scalar::Zero<Real>;
        Real b = Scalar::Zero<Real>;
        
        for( int i = 0; i < AmbDim; ++i )
        {
            a += (x[i]-y[i]) * (x[i]-y[i]);
            b += (x[i]+y[i]) * (x[i]+y[i]);
        }
        
//        return Scalar::Two<Real> * std::atan( std::sqrt(a/b) );
        
        return Scalar::Two<Real> * std::atan2( std::sqrt(a), std::sqrt(b) );
    }
    
    template<int AmbDim, typename Real>
    inline Real Angle( cptr<Real> x, cptr<Real> y )
    {
        Real a2 = Scalar::Zero<Real>;
        Real b2 = Scalar::Zero<Real>;

        for( int i = 0; i < AmbDim; ++i )
        {
            a2 += x[i] * x[i];
            b2 += y[i] * y[i];
        }
        
        const Real a_inv = InvSqrt(a2);
        const Real b_inv = InvSqrt(b2);
        
        Real u [AmbDim] = {};
        Real v [AmbDim] = {};
        
        for( int i = 0; i < AmbDim; ++i )
        {
            u[i] = x[i] * a_inv;
            v[i] = y[i] * a_inv;
        }
        
        return AngleBetweenUnitVectors<AmbDim>(&u[0],&v[0]);
    }
    
    template<typename T, typename I>
    inline constexpr T Factorial( I n )
    {
        T result ( 1 );
        
        for( T k = static_cast<T>(n); k > T(1); k -= T(1) )
        {
            result *= k;
        }
        
        return result;
    }
    
    template<typename T, typename I>
    inline constexpr T DoubleFactorial( I n )
    {
        T result ( 1 );
        
        for( T k = static_cast<T>(n); k > T(1); k -= T(2) )
        {
            result *= k;
        }
        
        return result;
    }
    
    
    template<typename T, typename I>
    inline constexpr T StandardSimplexVolume( I d )
    {
        return Inv( Factorial<T>(d) );
    }
    

    namespace Detail
    {
        template<typename Real>
        Real constexpr SqrtNewtonRaphson( const Real x, const Real curr, const Real prev )
        {
            static_assert(Scalar::RealQ<Real>,"");
            
            return (curr == prev)
                ? curr
                : SqrtNewtonRaphson(x, Scalar::Half<Real> * (curr + x / curr), curr);
        }
    }

    /*
    * Constexpr version of the square root
    * Return value:
    *   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
    *   - Otherwise, returns NaN
    * See https://stackoverflow.com/a/34134071/8248900.
    */
    template<typename Real>
    Real constexpr cSqrt( const Real x)
    {
        static_assert(Scalar::RealQ<Real>,"");
        
        return (x >= Scalar::Zero<Real> ) && (x <= std::numeric_limits<Real>::max())
            ? Detail::SqrtNewtonRaphson(x, x, Scalar::Zero<Real> )
            : std::numeric_limits<Real>::quiet_NaN();
    }
    
    
    template < typename T >
    inline constexpr T SphereVolume ( const Size_T n )
    {
        
        T result ( (n % 2 == 0) ? Scalar::Two<T> : Scalar::TwoPi<T> );
        
        for ( T k = n; k > Scalar::One<T>; k -= Scalar::Two<T> )
        {
            
            result *= Frac<T> ( Scalar::TwoPi<T>, k - 1);
        }
        
        return result;
    }
    
    template < typename T >
    inline constexpr T BallVolume ( const Size_T n )
    {
        
        T result ( (n % 2 == 0) ? Scalar::One<T> : Scalar::Two<T> );
        
        for ( T k = n; k > Scalar::One<T>; k -= Scalar::Two<T> )
        {
            
            result *= Frac<T> ( Scalar::TwoPi<T>, k );
        }
        
        return result;
    }
    
    template < typename T >
    inline constexpr T SOVolume ( const Size_T n )
    {
        
        T result ( Scalar::One<T> );
        
        for ( Size_T k = 1; k < n; ++k )
        {
            
            result *= SphereVolume<T>(k);
        }
        
        return result;
    }
    
    
    template < typename T >
    inline constexpr T GammaQuotient ( const T x, const T a )
    {
        // Computes Gamma(x+a)/Gamma(x) with considerably less risk of overflow than std::tgamma(x+a)/std::tgamma(x).

        static_assert(FloatQ<T>,"");
        
        static_assert( Scalar::RealQ<T>, "GammaQuotient is only implemented for real floating point types." );
        
        if( (x > Scalar::Zero<T>) && (a >= Scalar::Zero<T>) )
        {
            if( x <= Scalar::One<T> )
            {
                return std::tgamma(x+a) / std::tgamma(x);
            }
            else
            {
                T n;
                
                T r = modf( x, &n ) + Scalar::One<T>;
                
                // Adding 1 to guarantee that r > 0.
                
                // So r lies within [1,2).
                
                T result = std::tgamma(r+a) / std::tgamma(r);
                
                for ( T y = x; y --> r; )
                {
                    result += (a / y) * result;
                }
                
                return result;
            }
        }
        else
        {
            wprint("GammaQuotient is only implemented for positive x and nonnegative a. x = " + ToString(x) + ", a = " + ToString(a) + "." );
            return 1;
        }
    }
    
    
    template<typename Scal>
    inline constexpr Scal NextFloat( const Scal x )
    {
        TOOLS_MAKE_FP_STRICT()
        return std::nextafter( x, std::numeric_limits<Scal>::max() );
    }
    
    template<typename Scal>
    inline constexpr Scal PrevFloat( const Scal x )
    {
        TOOLS_MAKE_FP_STRICT()
        return std::nextafter( x, std::numeric_limits<Scal>::lowest() );
    }
    
    
    template<typename T>
    inline T ModDistance( const T L, const T i, const T j )
    {
        // Computes distance betreen mod(i,L) and mid(j,L) on the torus with arclength L. (Works also for integers.)
        
        static_assert( Scalar::RealQ<T>,"");

        if constexpr ( UnsignedIntQ<T> )
        {
            auto [a,b] = MinMax(i,j);
            
            const T delta = static_cast<T>(b - a);
            
            return Min( delta, static_cast<T>(L - delta) );
        }
        else
        {
            const T delta = static_cast<T>(Abs(j-i));
            
            return Min( delta, static_cast<T>(L - delta) );
        }
    }

    /*!@brief Checks whether `i` is contained in the interval defined by `a` and `b`.
     */
    template<
        bool left_closedQ = true, bool right_closedQ = false,
        typename T
    >
    inline bool InIntervalQ( const T i, const T a, const T b )
    {
        if constexpr( left_closedQ && !right_closedQ )
        {
            return (a <= i) && (i < b);
        }
        else if constexpr( left_closedQ && right_closedQ )
        {
            return (a <= i) && (i <= b);
        }
        else if constexpr( !left_closedQ && !right_closedQ )
        {
            return (a < i) && (i < b);
        }
        else if constexpr( !left_closedQ && right_closedQ )
        {
            return (a < i) && (i <= b);
        }
    }

}

 
