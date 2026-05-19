#pragma once

// Beware: At compile time all floating point rounding is towards nearest!!!

#include <cfenv>

#include "Tools.hpp"

// TODO: Read this:
// https://www.codeproject.com/Articles/1040839/Interval-arithmetic-using-round-to-nearest-mode-pa


// TODO: MinMax.
// TODO: integer power.
// TODO: operator/

// TODO: Things from the Scalar namespace, e.g., RealQ, ComplexQ, Prec etc.
// TODO: constants! (e.g. Pi)


namespace Tools
{
    enum struct RoundingPolicy : int
    {
        UseRoundingMode = 1,
        UseErrorTerm    = 0
    };
    
    [[nodiscard]] std::string ToString( const RoundingPolicy RP )
    {
        switch( RP )
        {
            case RoundingPolicy::UseRoundingMode:
            {
                return std::string("UseRoundingMode");
            }
            case RoundingPolicy::UseErrorTerm:
            {
                return std::string("UseErrorTerm");
            }
            default:
            {
                return std::string("Unknown");
            }
        }
    }
    
    class RoundingModeBarrier
    {
    protected:

        int prev_state;
        int state;

        
    public:
        
        RoundingModeBarrier() = delete;
        
        explicit RoundingModeBarrier( int state_ )
        :   prev_state( std::fegetround() )
        ,   state ( state_ )
        {
            std::fesetround( state );
            
//            TOOLS_DUMP( prev_state );
//            
//            TOOLS_DUMP( state );
//            
//            TOOLS_DUMP( std::fegetround() );
        }
        
        ~RoundingModeBarrier()
        {
            std::fesetround( prev_state );
            
//            TOOLS_DUMP( prev_state );
//            
//            TOOLS_DUMP( state );
//            
//            TOOLS_DUMP( std::fegetround() );
        }
        
    }; // class RoundingModeBarrier
    
    
    template<typename Real_, RoundingPolicy RP_>
    class Interval;
    
    template<typename Real_, RoundingPolicy RP_>
    class Interval;
    
    
    // From Hida, Li, Bailey - Library for Double-Double and Quad-Double Arithmetic
    template<typename Real>
    TOOLS_FORCE_INLINE std::pair<Real,Real> TwoAdd( const Real a, const Real b )
    {
        const Real s = a + b;
        const Real v = s - b;
        const Real e = (a - (s - v)) + (b-v);
        
        return std::pair(s,e);
    }
    
    // From Hida, Li, Bailey - Library for Double-Double and Quad-Double Arithmetic
    template<typename Real>
    TOOLS_FORCE_INLINE std::pair<Real,Real> TwoMulFMA( const Real a, const Real b )
    {
        const Real p = a * b;
        const Real e = std::fma(a,b,-p);
        
        return std::pair(p,e);
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE Real RoundUp( const Real x, const bool increaseQ )
    {
        return std::nextafter( x, increaseQ ? std::numeric_limits<Real>::max() : x );
    }
    
    template<typename Real, RoundingPolicy RP = RoundingPolicy::UseRoundingMode>
    class Rounding_T
    {
        static_assert(FloatQ<Real>,"");
        
    protected:

        static void RoundingModeWarning( cref<std::string> tag )
        {
            #ifdef CHECK_ROUNDING_MODE
          
            if( std::fegetround() != FE_UPWARD )
            {
                eprint(ClassName()+"::" + tag + ": Rounding mode not set to FE_UPWARD.");
            }
            #endif
        }

        TOOLS_FORCE_INLINE static Real AddRoundUp( const Real a, const Real b )
        {
            if constexpr ( RP == RoundingPolicy::UseRoundingMode )
            {
                    RoundingModeWarning("AddRoundUp");

                    return a + b;
            }
            else
            {
                const auto [p, e] = TwoAdd( a, b );
                
                return RoundUp( p, e > Real(0) );
            }
        }
        
        TOOLS_FORCE_INLINE static Real MulRoundUp( const Real a, const Real b )
        {
            if constexpr ( RP == RoundingPolicy::UseRoundingMode)
            {
                RoundingModeWarning("MulRoundUp");
                return a * b;
            }
            else
            {
                const auto [p, e] = TwoMulFMA( a, b );
                
                return RoundUp( p, e > Real(0) );
            }
        }
        
        static std::string ClassName()
        {
            return std::string("Rounding_T<") + TypeName<Real> + ", " + ToString(RP) + ">";
        }
        
    }; // class Rounding_T
    
    
    template<typename Real_, RoundingPolicy RP_ = RoundingPolicy::UseRoundingMode>
    class Singleton : Rounding_T<Real_,RP_>
    {
        static_assert(FloatQ<Real_>,"");
        
    public:
        
        using Real = Real_;
        
        static constexpr RoundingPolicy RP = RP_;

        using Base_T = Rounding_T<Real,RP>;
        
        friend class Interval<Real,RP>;
        
        using S_T = Singleton<Real,RP>;
        using I_T = Interval <Real,RP>;
        
    protected:
        
        using Base_T::AddRoundUp;
        using Base_T::MulRoundUp;
        using Base_T::RoundingModeWarning;
        
        Real x;
        
    public:
        
        Singleton()
        : x{ 0 }
        {}
        
        explicit Singleton( const Real & x_ )
        : x{ x_ }
        {}
        
        ~Singleton() = default;
        
        // Copy constructor
        Singleton( const Singleton & other )
        :   x { other.x }
        {}
        
        friend void swap (Singleton & s, Singleton & t ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
            
            swap( s.x, t.x );
        }
        
        // (Copy-)assignment operator
        Singleton & operator=( Singleton other )
        {
            swap( *this, other );
            
            return *this;
        }
        
        // Move constructor
        Singleton( Singleton && other ) noexcept
        {
            swap(*this, other);
        }
        
        Real Lower() const
        {
            return x;
        }
        
        Real Upper() const
        {
            return x;
        }
    
        
//######################################################
//##                  Univariate                      ##
//######################################################
        
    public:
        
        TOOLS_FORCE_INLINE friend S_T operator+( const S_T s )
        {
            return S_T{s.x};
        }
        
        TOOLS_FORCE_INLINE friend S_T operator-( const S_T s )
        {
            return S_T{-s.x};
        }
        
        TOOLS_FORCE_INLINE friend S_T Abs( const S_T s )
        {
            return S_T{Abs(s.x)};
        }
        
        TOOLS_FORCE_INLINE friend I_T Square( const S_T s )
        {
            return I_T::Create( MulRoundUp(- s.x, s.x),  MulRoundUp(s.x, s.x) );
        }

        TOOLS_FORCE_INLINE friend S_T Ramp( const S_T s )
        {
            return S_T{Ramp(s.x)};
        }
        
        template<typename R = int>
        TOOLS_FORCE_INLINE friend R Sign( const S_T s )
        {
            return Sign<R>(s.x);
        }
        
        
//######################################################
//##                   Bivariate                      ##
//######################################################
    
    public:
        
        TOOLS_FORCE_INLINE friend I_T operator+( const S_T s, const S_T t )
        {
            return I_T::Create( AddRoundUp(-s.x,-t.x), AddRoundUp( s.x, t.x) );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator-( const S_T s, const S_T t )
        {
            return I_T::Create( AddRoundUp(-s.x, t.x), AddRoundUp( s.x,-t.x) );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator*( const S_T s, const S_T t )
        {
            return I_T::Create( MulRoundUp(-s.x, t.x), MulRoundUp( s.x, t.x) );
        }
        
        
        TOOLS_FORCE_INLINE friend S_T Max( const S_T s, const S_T t )
        {
            return S_T( Max( s.x, t.x ) );
        }
        
        TOOLS_FORCE_INLINE friend S_T Min( const S_T s, const S_T t )
        {
            return S_T( Min( s.x, t.x ) );
        }
        
        
//######################################################
//##                   Trivariate                     ##
//######################################################
        
        
        TOOLS_FORCE_INLINE friend I_T fma( const S_T r, const S_T s, const S_T t )
        {
            RoundingModeWarning("fma (SSS)");
            
            return I_T::Create( std::fma( r.x, -s.x, -t.x ), std::fma( r.x, s.x, t.x ) );
        }
        
        static std::string ClassName()
        {
            return std::string("Singleton<") + TypeName<Real> + ">";
        }
        
    }; // class Singleton
    
    
    
    
    
    template<typename Real_, RoundingPolicy RP_ = RoundingPolicy::UseRoundingMode>
    class Interval : Rounding_T<Real_,RP_>
    {
        static_assert(FloatQ<Real_>,"");
        
    public:
        
        using Real = Real_;

        static constexpr RoundingPolicy RP = RP_;
        
        using Base_T = Rounding_T<Real,RP>;
        
        friend class Singleton<Real,RP>;
        
        using R_T = Real;
        using I_T = Interval<Real,RP>;
        using S_T = Singleton<Real,RP>;
        
    protected:
        
        using Base_T::AddRoundUp;
        using Base_T::MulRoundUp;
        using Base_T::RoundingModeWarning;
        
        Real a;
        Real b;
        
    public:
        
        Interval()
        : a{ 0 }
        , b{ 0 }
        {}
        
        explicit Interval( cref<R_T> x )
        : a{ -x }
        , b{  x }
        {}
        
        explicit Interval( cref<S_T> s )
        : a{ -s.x }
        , b{  s.x }
        {}
        
        
        // Copy constructor
        Interval( cref<I_T> J )
        :   a { J.a }
        ,   b { J.b }
        {}
        
        explicit Interval( const R_T lower_, const R_T upper_ )
        : a{ -Min(lower_,upper_) }
        , b{  Max(lower_,upper_) }
        {}
        
        ~Interval() = default;
        
    
//#if defined(LTEMPLATE_H)
//        
//        explicit Interval( cref<mma::TensorRef<Real>> A )
//        : Interval{ A.data()[0], A.data()[1]  }
//        {}
//        
//        mma::TensorRef<Real> friend to_MTensorRef( const I_T I )
//        {
//            auto R = mma::makeVector<Real>( 2 );
//            
//            R.data()[0] = -I.a;
//            R.data()[1] =  I.b;
//            
//            return R;
//        }
//#endif

        
        
        static Interval Create( const R_T a, const R_T b )
        {
#if defined(INTERVAL_DEBUG)
            if( -a > b )
            {
                eprint("Create: -a > b");
                TOOLS_DUMP(-a);
                TOOLS_DUMP(b);
            }
#endif
            Interval K;
            K.a = a;
            K.b = b;
            
            return K;
        }
        
        friend void swap( I_T & I, I_T & J ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
            
            swap( I.a, J.a );
            swap( I.b, J.b );
        }
        
        // (Copy-)assignment operator
        Interval & operator=( I_T other )
        {
            swap( *this, other );
            
            return *this;
        }
        
        // Move constructor
        Interval( I_T && other ) noexcept
        {
            swap(*this, other);
        }
      
        
//######################################################
//##                    Access                        ##
//######################################################
    
    public:

        Real Upper() const
        {
            return b;
        }
        
        Real Lower() const
        {
            return -a;
        }
        
        Real Width() const
        {
            return (a + b);
        }
        
        Real EpsRelativeWidth() const
        {
            return ( (a + b) / Max(Abs(a),Abs(b)) ) / Scalar::eps<Real>;
        }
        
//######################################################
//##                   Booleans                       ##
//######################################################

    public:
        
        TOOLS_FORCE_INLINE bool SingletonQ() const
        {
            return (-a==b);
        }
        
        TOOLS_FORCE_INLINE bool ContainsZeroQ() const
        {
            return (b >= Real(0)) && (a >= Real(0));
        }
        
        TOOLS_FORCE_INLINE bool ContainsQ( const R_T x ) const
        {
            return (b >= x) && (-a <= x);
        }
        
        TOOLS_FORCE_INLINE bool NotContainsQ( const R_T x ) const
        {
            return (b < x) || (-a > x);
        }
        
        TOOLS_FORCE_INLINE bool FullyNegativeQ() const
        {
            return b < Real(0);
        }
        
        TOOLS_FORCE_INLINE bool FullyPositiveQ() const
        {
            return a < Real(0);
        }
        

        TOOLS_FORCE_INLINE friend bool operator<( const I_T I, const I_T J )
        {
            return I.b < -J.a;
        }
        
        TOOLS_FORCE_INLINE friend bool operator>( const I_T I, const I_T J )
        {
            return -I.a > J.b;
        }
        
        
        
        TOOLS_FORCE_INLINE friend bool operator<( const I_T I, const R_T x )
        {
            return I.b < x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator<( const I_T I, const S_T s )
        {
            return I < s.x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator>( const R_T x, const I_T I )
        {
            return I < x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator>( const S_T s, const I_T I )
        {
            return I < s.x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator>( const I_T I, const R_T x )
        {
            return -I.a > x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator>( const I_T I, const S_T s )
        {
            return I > s.x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator<( const R_T x, const I_T I )
        {
            return I > x;
        }
        
        TOOLS_FORCE_INLINE friend bool operator<( const S_T s, const I_T I )
        {
            return I > s.x;
        }
        
//######################################################
//##                  Univariate                      ##
//######################################################
        
    public:
        
        TOOLS_FORCE_INLINE friend I_T operator+( const I_T I )
        {
            return Create( I.a, I.b );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator-( const I_T I )
        {
            return Create( I.b, I.a );
        }
        
        TOOLS_FORCE_INLINE friend I_T Abs( const I_T I )
        {
            const Real A = Abs(I.a);
            const Real B = Abs(I.b);
            
            return Create( I.ContainsZeroQ() ? 0 : Min(A,B), Max(A,B) );
        }
        
        TOOLS_FORCE_INLINE friend I_T Square( const I_T I )
        {
            const Real A = MulRoundUp( I.a, I.a );
            const Real B = MulRoundUp( I.b, I.b );
            
            return Create( I.ContainsZeroQ() ? 0 : Min(A,B), Max(A,B) );
        }

        TOOLS_FORCE_INLINE friend I_T Ramp( const I_T I )
        {
            return Create( Min(Scalar::Zero<Real>,I.a), Max(Scalar::Zero<Real>,I.b) );
        }
        
        template<typename R = int>
        TOOLS_FORCE_INLINE friend R Sign( const I_T I )
        {
            constexpr Real zero = 0;
            
            return static_cast<R>( (I > zero) - (I < zero) );
        }
        
        

        
//######################################################
//##                   Bivariate                      ##
//######################################################
    
    public:
        
        TOOLS_FORCE_INLINE friend I_T operator+( const I_T I, const I_T J )
        {
            return Create( AddRoundUp( I.a, J.a ), AddRoundUp( I.b, J.b ) );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator+( const I_T I, const R_T x )
        {
            return Create( AddRoundUp( I.a, -x ), AddRoundUp( I.b, x ) );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator+( const I_T I, const S_T s )
        {
            return I + s.x;
        }
        
        
        TOOLS_FORCE_INLINE friend I_T operator+( const R_T x, const I_T I )
        {
            return I + x;
        }
        
        TOOLS_FORCE_INLINE friend I_T operator+( const S_T s, const I_T I )
        {
            return I + s.x;
        }
        
        
        
        TOOLS_FORCE_INLINE friend I_T operator-( const I_T I, const I_T J )
        {
            return Create( AddRoundUp( I.a, J.b ), AddRoundUp( I.b, J.a ) );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator-( const I_T I, const R_T x )
        {
            return Create( AddRoundUp( I.a, x ) , AddRoundUp( I.b, -x ) );
        }
        

        TOOLS_FORCE_INLINE friend I_T operator-( const I_T I, const S_T s )
        {
            return I + s.x;
        }
        

        TOOLS_FORCE_INLINE friend I_T operator-( const R_T x, const I_T I )
        {
            return Create( AddRoundUp( -x, I.b ) , AddRoundUp( x, I.a ) );
            
        }

        TOOLS_FORCE_INLINE friend I_T operator-( const S_T s, const I_T J )
        {
            return s.x - J;
        }

        
        TOOLS_FORCE_INLINE friend I_T mult_II_pos_pos( const I_T I, const I_T J )
        {
            // I.Lower() > 0, I.Upper() > 0, J.Lower() > 0, J.Upper() > 0
            // [ I.Lower() * J.Lower(), I.Upper() * J.Upper() ]
            
            return Create( MulRoundUp(I.a, -J.a), MulRoundUp(I.b, J.b) );
        }
        
        TOOLS_FORCE_INLINE friend I_T mult_II_switch( const I_T I, const I_T J )
        {
            switch( Sign(I) )
            {
                case 1:
                {
                    switch( Sign(J) )
                    {
                        case 1:
                        {
                            return mult_II_pos_pos( I, J );
                        }
                        case -1:
                        {
                            // I.Lower() > 0, I.Upper() > 0, J.Lower() < 0, J.Upper() < 0
                            // [ I.Upper() * J.Lower(), I.Lower() * J.Upper() ]

                            return Create( MulRoundUp(I.b,J.a), MulRoundUp(-I.a,J.b) );
                        }
                        case 0:
                        {
                            // I.Lower() > 0, I.Upper() > 0, J.Lower() <= 0, J.Upper() >= 0
                            // [ I.Upper() * J.Lower(), I.Upper() * J.Upper() ]

                            return Create( MulRoundUp(I.b,J.a), MulRoundUp(I.b,J.b) );
                        }
                    }
                }
                case -1:
                {
                    switch( Sign(J) )
                    {
                        case 1:
                        {
                            // I.Lower() < 0, I.Upper() < 0, J.Lower() > 0, J.Upper() > 0
                            // [ I.Lower() * J.Upper(), I.Upper() * J.Lower() ]
                            
                            return Create( MulRoundUp(I.a,J.b), MulRoundUp(I.b,-J.a) );
                        }
                        case -1:
                        {
                            // I.Lower() < 0, I.Upper() < 0, J.Lower() < 0, J.Upper() < 0
                            // [ I.Upper() * J.Upper(), I.Lower() * J.Lower() ]
                            
                            return Create( MulRoundUp(-I.b,J.b), MulRoundUp(I.a,J.a) );
                        }
                        case 0:
                        {
                            // I.Lower() < 0, I.Upper() < 0, J.Lower() <= 0, J.Upper() >= 0
                            // [ I.Lower() * J.Upper(), I.Lower() * J.Lower() ]
                            
                            return Create( MulRoundUp(I.a,J.b), MulRoundUp(I.a,J.a) );
                        }
                    }
                }
                case 0:
                {
                    switch( Sign(J) )
                    {
                        case 1:
                        {
                            // I.Lower() <= 0, I.Upper() >= 0, J.Lower() > 0, J.Upper() > 0
                            // [ I.Lower() * J.Upper(), I.Upper() * J.Upper() ]
                            
                            return Create( MulRoundUp(I.a,J.b), MulRoundUp(I.b,J.b) );
                        }
                        case -1:
                        {
                            // I.Lower() <= 0, I.Upper() >= 0, J.Lower() < 0, J.Upper() < 0
                            // [ I.Upper() * J.Lower(), I.Lower() * J.Lower() ]
                            
                            return Create( MulRoundUp(I.b,J.a), MulRoundUp(I.a,J.a) );
                        }
                        case 0:
                        {
                            // I.Lower() <= 0, I.Upper() => 0, J.Lower() <= 0, J.Upper() => 0
                            // [ Min(I.Lower() * J.Upper(), I.Upper() * J.Lower(),
                            //   Max(I.Lower() * J.Lower(), I.Upper() * J.Upper() ]
                            
                            return Create(
                                Max( MulRoundUp(I.a,J.b), MulRoundUp(I.b,J.a) ),
                                Max( MulRoundUp(I.a,J.a), MulRoundUp(I.b,J.b) )
                            );
                        }
                    }
                }
                    
            } // switch( Sign(I) )
            
            return I_T();
        }
        
        
        TOOLS_FORCE_INLINE friend I_T mult_RI_switch( const R_T x, const I_T J )
        {
            switch( Sign(x) )
            {
                case 1:
                {
                    return Create( MulRoundUp(x,J.a), MulRoundUp(x,J.b) );
                }
                case -1:
                {
                    return Create( MulRoundUp(-x,J.b), MulRoundUp(-x,J.a) );
                }
                case 0:
                {
                    return Create( 0, 0 );
                }
                    
            } // switch( Sign(x) )
            
            return I_T();
        }
        
        
        
        TOOLS_FORCE_INLINE friend I_T mul_II_bruteforce( const I_T I, const I_T J )
        {
            // 8(!) multiplications, but apparently, they can be vectorized.
            
            const Real A = Max(
                Max( MulRoundUp( -I.a, J.a), MulRoundUp(  I.a, J.b) ),
                Max( MulRoundUp(  I.b, J.a), MulRoundUp( -I.b, J.b) )
            );
            
            const Real B = Max(
                Max( MulRoundUp(  I.a, J.a), MulRoundUp( -I.a, J.b) ),
                Max( MulRoundUp( -I.b, J.a), MulRoundUp(  I.b, J.b) )
            );
            
            return Create( A, B );
        }
        
        TOOLS_FORCE_INLINE friend I_T mul_IR_bruteforce( const I_T I, const R_T x )
        {
            return Create(
                Max( MulRoundUp(  I.a, x ), MulRoundUp( -I.b, x ) ),
                Max( MulRoundUp( -I.a, x ), MulRoundUp(  I.b, x ) )
            );
        }
        
        TOOLS_FORCE_INLINE friend I_T mul_II_bruteforce2( const I_T I, const I_T J )
        {
            // Advantage: No switching of rounding mode needed.
            // Is slower than mul_II_bruteforce, though.
            // For some reason, clang does not want to vectorize this.
            
            const auto [aa, e_aa] = TwoMulFMA( I.a, J.a );
            const auto [ab, e_ab] = TwoMulFMA( I.a, J.b );
            const auto [ba, e_ba] = TwoMulFMA( I.b, J.a );
            const auto [bb, e_bb] = TwoMulFMA( I.b, J.b );

            const Real A = Max(
                Max( RoundUp( -aa, e_aa < 0 ), RoundUp(  ab, e_ab > 0 ) ),
                Max( RoundUp(  ba, e_ba > 0 ), RoundUp( -bb, e_bb < 0 ) )
            );
            
            const Real B = Max(
                Max( RoundUp(  aa, e_aa > 0 ), RoundUp( -ab, e_ab < 0 ) ),
                Max( RoundUp( -ba, e_ba < 0 ), RoundUp(  bb, e_bb > 0 ) )
            );
            
            return Create( A, B );
        }
        
        TOOLS_FORCE_INLINE friend I_T mul_IR_bruteforce2( const I_T I, const R_T x )
        {
            const auto [ax, e_ax] = TwoMulFMA( I.a, x );
            const auto [bx, e_bx] = TwoMulFMA( I.b, x );

            const Real A = Max( RoundUp(  ax, e_ax > 0 ), RoundUp( -bx, e_bx < 0 ) );
            
            const Real B = Max( RoundUp( -ax, e_ax < 0 ), RoundUp(  bx, e_bx > 0 ) );
            
            return Create( A, B );
            
        }
        
        TOOLS_FORCE_INLINE friend I_T operator*( const I_T I, const I_T J )
        {
            return mul_II_bruteforce( I, J );
            
//            return mul_II_bruteforce2( I, J );
            
//            return mult_II_switch( I, J );
        }
                                          
        TOOLS_FORCE_INLINE friend I_T operator*( const I_T I, const R_T x )
        {
            return mul_IR_bruteforce( I, x );
            
//            return mul_IR_bruteforce2( I, x );
                                
//            return mult_RI_switch( x, I );
        }
        
        TOOLS_FORCE_INLINE friend I_T operator*( const I_T I, const S_T s )
        {
            return I * s.x;
        }
        
        
        TOOLS_FORCE_INLINE friend I_T operator*( const R_T x, const I_T I )
        {
            return I * x;
        }
        
        TOOLS_FORCE_INLINE friend I_T operator*( const S_T s, const I_T I )
        {
            return I * s.x;
        }
        
        
        
        
        
        TOOLS_FORCE_INLINE friend I_T Max( const I_T I, const I_T J )
        {
            return Create( Min(I.a,J.a), Max(I.b,J.b) );
        }
        

        TOOLS_FORCE_INLINE friend I_T Max( const I_T I, const R_T x )
        {
            return Create( Min(I.a,-x), Max(I.b,x) );
        }
        
        TOOLS_FORCE_INLINE friend I_T Max( const I_T I, const S_T s )
        {
            return Max(I,s.x);
        }
        
        
        TOOLS_FORCE_INLINE friend I_T Max( const R_T x, const I_T I )
        {
            return Create( Min(-x,I.a), Max(x,I.b) );
        }
        
        TOOLS_FORCE_INLINE friend I_T Max( const S_T s, const I_T I )
        {
            return Max(I,s.x);
        }
        
        
        TOOLS_FORCE_INLINE friend I_T Min( const I_T I, const I_T J )
        {
            return Create( Max(I.a,J.a), Min(I.b,J.b) );
        }
        
        
        TOOLS_FORCE_INLINE friend I_T Min( const I_T I, const R_T x )
        {
            return Create( Max(I.a,-x), Min(I.b,x) );
        }
        
        TOOLS_FORCE_INLINE friend I_T Min( const I_T I, const S_T s )
        {
            return Min(I,s.x);
        }

        TOOLS_FORCE_INLINE friend I_T Min( const R_T x, const I_T I )
        {
            return Min(I,x);
        }
        
        TOOLS_FORCE_INLINE friend I_T Min( const S_T s, const I_T I )
        {
            return Min(I,s.x);
        }
        
//######################################################
//##                   Trivariate                      ##
//######################################################
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T I, const I_T J, const I_T K )
        {
            RoundingModeWarning("fma (III)");

            const Real A = Max(
                Max( std::fma( -I.a, J.a, K.a ), std::fma(  I.a, J.b, K.a) ),
                Max( std::fma(  I.b, J.a, K.a ), std::fma( -I.b, J.b, K.a) )
            );
            
            const Real B = Max(
                Max( std::fma(  I.a, J.a, K.b ), std::fma( -I.a, J.b, K.b ) ),
                Max( std::fma( -I.b, J.a, K.b ), std::fma(  I.b, J.b, K.b ) )
            );
            
            return Create(A,B);
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T I, const I_T J, const R_T z )
        {
            RoundingModeWarning("fma (IIR)");

            const Real A = Max(
                Max( std::fma( -I.a, J.a, -z ), std::fma(  I.a, J.b, -z) ),
                Max( std::fma(  I.b, J.a, -z ), std::fma( -I.b, J.b, -z) )
            );
            
            const Real B = Max(
                Max( std::fma(  I.a, J.a, z ), std::fma( -I.a, J.b, z ) ),
                Max( std::fma( -I.b, J.a, z ), std::fma(  I.b, J.b, z ) )
            );
            
            return Create(A,B);
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T I, const I_T J, const S_T t )
        {
            return fma(I,J,t.x);
        }

        
        TOOLS_FORCE_INLINE friend I_T fma( const R_T x, const I_T J, const I_T K )
        {
            RoundingModeWarning("fma (RII)");

            const Real A = Max( std::fma(  x, J.a, K.a ), std::fma( -x, J.b, K.a ) );
            const Real B = Max( std::fma( -x, J.a, K.b ), std::fma(  x, J.b, K.b ) );
            
            return Create(A,B);
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const S_T s, const I_T J, const I_T K )
        {
            return fma( s.x, J, K );
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T J, const R_T x, const I_T K )
        {
            return fma( x, J, K );
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T J, const S_T s, const I_T K )
        {
            return fma( s.x, J, K );
        }
        
        
        TOOLS_FORCE_INLINE friend I_T fma( const R_T x, const R_T y, const I_T K )
        {
            RoundingModeWarning("fma (RRI)");
            
            return Create(std::fma( -x, y, K.a ), std::fma( x, y, K.b ) );
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const S_T s, const S_T t, const I_T K )
        {
            return fma( s.x, t.x, K );
        }
        
        
        TOOLS_FORCE_INLINE friend I_T fma( const R_T x, const I_T I, const R_T z )
        {
            RoundingModeWarning("fma (RIR)");
            
            return Create (
                Max( std::fma( x,  I.a, -z ), std::fma( x, -I.b, -z ) ),
                Max( std::fma( x, -I.a,  z ), std::fma( x,  I.b,  z ) )
            );
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const S_T r, const I_T I, const S_T t )
        {
            return fma( r.x, I, t.x );
        }
        
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T I, const R_T x, const R_T z )
        {
            return fma( x, I, z );
        }
        
        TOOLS_FORCE_INLINE friend I_T fma( const I_T I, const S_T r, const S_T t )
        {
            return fma( r.x, I, t.x );
        }
        
        
        static std::string ClassName()
        {
            return std::string("Interval<") + TypeName<Real> + ", " + ToString(RP) + ">";
        }
        
        
    }; // class Interval
    

    template<typename Real, RoundingPolicy RP >
    [[nodiscard]] std::string ToString( const Singleton<Real,RP> s )
    {
        return std::string("{ ") + ToString(s.x) + " }";
    }
    
    template<typename Real, RoundingPolicy RP >
    [[nodiscard]] std::string ToString( const Interval<Real,RP> I )
    {
        return std::string("[ ") + ToString(I.Lower()) + ", " + ToString(I.Upper()) + " ]";
    }
    
    
    
    template<int D, bool fmaQ = true, typename c_T, typename x_T>
    decltype( c_T(1) * x_T(1) ) EvaluatePolynomial_Horner( cptr<c_T> c, const x_T x )
    {
        using std::fma;
        
        using oux_T = decltype( c_T(1) * x_T(1) );
        
        if( D <= int(0) )
        {
            return oux_T{c[0]};
        }
        
        if constexpr ( fmaQ )
        {
            oux_T r = fma( c[D], x, c[D-int(1)] );
            
            for( int i = D-int(1); i --> int(0);  )
            {
                
                r = fma( r, x, c[i] );
            }
            
            return r;
        }
        else
        {
            oux_T r = c[D] * x + c[D-int(1)];
            
            for( int i = D-int(1); i --> int(0);  )
            {
                
                r = r * x + c[i];
            }
            
            return r;
        }
        
    }
    
} // namespace Tools



#include "Oriented2D.hpp"
#include "Oriented3D.hpp"
