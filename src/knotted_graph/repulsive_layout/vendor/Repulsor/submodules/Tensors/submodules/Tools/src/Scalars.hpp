#include <complex>

namespace Tools
{
    
    using Real32  = float;
    using Real64  = double;
//    using Real128 = long double;
    
    using Complex32  = std::complex<Real32>;
    using Complex64  = std::complex<Real64>;
//    using Complex128 = std::complex<Real128>;
    
    template<typename R> using Cplx = std::complex<R>;
    
    template<typename R>
    TOOLS_FORCE_INLINE constexpr vec_T<2,R> & to_vec_T( cref<Cplx<R>> z )
    {
        static_assert(sizeof(vec_T<2,R>) == sizeof(Cplx<R>), "");
        
        return *reinterpret_cast<const vec_T<2,R> *>(&z);
    }
    
    template<typename R>
    TOOLS_FORCE_INLINE vec_T<2,R> & to_vec_T( mref<Cplx<R>> z )
    {
        static_assert(sizeof(vec_T<2,R>) == sizeof(Cplx<R>), "");
        
        return *reinterpret_cast<vec_T<2,R> *>(&z);
    }
    
    template<typename R>
    TOOLS_FORCE_INLINE constexpr Cplx<R> to_Cplx( cref<vec_T<2,R>> Z )
    {
        static_assert(sizeof(vec_T<2,R>) == sizeof(Cplx<R>), "");
        
        Cplx<R> z;
        
        *reinterpret_cast<const vec_T<2,R> *>(&z) = Z;
    
        return z;
    }
    
    
    using std::conj;
    using std::real;
    using std::imag;
    
    template<typename S, typename T>
    constexpr bool SameQ = std::is_same_v<S,T>;
    
    template<typename T>
    constexpr bool VectorizableQ = vec_enabledQ && ( SameQ<T,Real32> || SameQ<T,Real64> || SameQ<T,Int16> || SameQ<T, Int32> || SameQ<T,Int64> || SameQ<T,UInt16> ||SameQ<T,UInt32> || SameQ<T,UInt64> );
    
    template<typename T>
    constexpr bool MatrixizableQ = mat_enabledQ
        && (
            SameQ<T,Real32> || SameQ<T,Real64>
            || SameQ<T, Int16> || SameQ<T, Int32> || SameQ<T, Int64>
            || SameQ<T,UInt16> || SameQ<T,UInt32> || SameQ<T,UInt64>
        );
    
    namespace Scalar
    {
        enum class Flag : Int8
        {
            Generic =  2,
            Plus    =  1,
            Minus   = -1,
            Zero    =  0
        };
        
        template<typename T>
        [[nodiscard]] TOOLS_FORCE_INLINE constexpr Flag ToFlag( const T x )
        {
            if( x == T(0) )
            {
                return Flag::Zero;
            }
            else if( x == T(1) )
            {
                return Flag::Plus;
            }
            else if( x == T(-1) )
            {
                return Flag::Minus;
            }
            else
            {
                return Flag::Generic;
            }
        }
        
        template<typename T>
        [[nodiscard]] TOOLS_FORCE_INLINE std::string ToFlagString( const T x )
        {
            if( x == T(0) )
            {
                return "Tools::Scalar::Flag::Zero";
            }
            else if( x == T(1) )
            {
                return "Tools::Scalar::Flag::Plus";
            }
            else if( x == T(-1) )
            {
                return "Tools::Scalar::Flag::Minus";
            }
            else
            {
                return "Tools::Scalar::Flag::Generic";
            }
        }
        
    } // namespace Scalar
    
    
    
    
    [[nodiscard]] TOOLS_FORCE_INLINE std::string ToString( const Scalar::Flag flag )
    {
        switch( flag )
        {
            case Scalar::Flag::Plus:
            {
                return "+";
            }
            case Scalar::Flag::Zero:
            {
                return "0";
            }
            case Scalar::Flag::Minus:
            {
                return "-";
            }
            case Scalar::Flag::Generic:
            {
                return "G";
            }
            default:
            {
                eprint(std::string("ToString: flag value ") + ToString(ToUnderlying(flag)) + " is invalid.");
                return "X";
            }
        }
    }
    
    namespace Scalar
    {
        template<> constexpr bool ComplexQ<Real32>     = false;
        template<> constexpr bool ComplexQ<Real64>     = false;
        //        template<> constexpr bool ComplexQ<Real128>    = false;
        template<> constexpr bool ComplexQ<Complex32>  = true;
        template<> constexpr bool ComplexQ<Complex64>  = true;
        //        template<> constexpr bool ComplexQ<Complex128> = true;
        
        
        template<> constexpr bool RealQ<Real32 >       = true;
        template<> constexpr bool RealQ<Real64 >       = true;
        
        template<> constexpr bool RealQ<bool   >       = true;
        template<> constexpr bool RealQ<Int8   >       = true;
        template<> constexpr bool RealQ<Int16  >       = true;
        template<> constexpr bool RealQ<Int32  >       = true;
        template<> constexpr bool RealQ<Int64  >       = true;
        template<> constexpr bool RealQ<UInt8  >       = true;
        template<> constexpr bool RealQ<UInt16 >       = true;
        template<> constexpr bool RealQ<UInt32 >       = true;
        template<> constexpr bool RealQ<UInt64 >       = true;

        
        //        template<> constexpr bool RealQ<Real128>       = true;
        template<> constexpr bool RealQ<Complex32 >    = false;
        template<> constexpr bool RealQ<Complex64 >    = false;
        //        template<> constexpr bool RealQ<Complex128>    = false;
        
        template<typename T> constexpr bool ScalarQ    = false;
        
        template<> constexpr bool ScalarQ<Real32 >     = true;
        template<> constexpr bool ScalarQ<Real64 >     = true;
        //        template<> constexpr bool ScalarQ<Real128>     = true;
        template<> constexpr bool ScalarQ<Complex32 >  = true;
        template<> constexpr bool ScalarQ<Complex64 >  = true;
        //        template<> constexpr bool ScalarQ<Complex128>  = true;
        
        
        template<> constexpr bool ScalarQ<bool>     = true;
        template<> constexpr bool ScalarQ<Int8>     = true;
        template<> constexpr bool ScalarQ<Int16>    = true;
        template<> constexpr bool ScalarQ<Int32>    = true;
        template<> constexpr bool ScalarQ<Int64>    = true;
        template<> constexpr bool ScalarQ<UInt8>    = true;
        template<> constexpr bool ScalarQ<UInt16>   = true;
        template<> constexpr bool ScalarQ<UInt32>   = true;
        template<> constexpr bool ScalarQ<UInt64>   = true;
        
        template<typename T> constexpr bool FloatQ = false;
        
        template<> constexpr bool FloatQ<Real32 >     = true;
        template<> constexpr bool FloatQ<Real64 >     = true;
        template<> constexpr bool FloatQ<Complex32 >  = true;
        template<> constexpr bool FloatQ<Complex64 >  = true;
        
        template<typename T> constexpr int Prec       = 0;
        
        template<> constexpr int Prec<Real32 >     = sizeof(Real32) * 8;
        template<> constexpr int Prec<Real64 >     = sizeof(Real64) * 8;
        template<> constexpr int Prec<Complex32 >  = sizeof(Real32) * 8;
        template<> constexpr int Prec<Complex64 >  = sizeof(Real64) * 8;
        
        
        template<typename T>
        using Real =
        std::conditional_t<Prec<T> ==  32,  Real32,
        std::conditional_t<Prec<T> ==  64,  Real64,
        T>>;
        
        template<typename T>
        using Complex =
        std::conditional_t<Prec<T> ==  32,  Complex32,
        std::conditional_t<Prec<T> ==  64,  Complex64,
        T>>;
        
        template<typename T> constexpr Scalar::Real<T> Zero  {0};
        template<typename T> constexpr Scalar::Real<T> One   {1};
        template<typename T> constexpr Scalar::Real<T> Two   {2};
        template<typename T> constexpr Scalar::Real<T> Three {3};
        template<typename T> constexpr Scalar::Real<T> Four  {4};
        template<typename T> constexpr Scalar::Real<T> Fife  {5};
        template<typename T> constexpr Scalar::Real<T> Six   {6};
        template<typename T> constexpr Scalar::Real<T> Half    = One<T>/Two<T>;
        template<typename T> constexpr Scalar::Real<T> Third   = One<T>/Three<T>;
        template<typename T> constexpr Scalar::Real<T> Quarter = One<T>/Four<T>;
        template<typename T> constexpr Scalar::Real<T> Fourth  = One<T>/Four<T>;
        template<typename T> constexpr Scalar::Real<T> Fifth   = One<T>/Fife<T>;
        template<typename T> constexpr Scalar::Real<T> Sixth   = One<T>/Six<T>;
        
        
        template<typename T> constexpr Scalar::Complex<T> I {0,1};
        
        template<typename T> constexpr Scalar::Real<T> Pi = static_cast<Scalar::Real<T>>(3.141592653589793238462643383279502884197);
        
        template<typename T> constexpr Scalar::Real<T> TwoPi    = Scalar::Two<T> * Scalar::Pi<T>;
        template<typename T> constexpr Scalar::Real<T> TwoPiInv = Scalar::One<T> / ( Scalar::Two<T> * Scalar::Pi<T> );
        
        template<typename T> constexpr Scalar::Real<T> eps      = std::numeric_limits<Scalar::Real<T>>::epsilon();
        
        template<typename T> constexpr Scalar::Real<T> eta      = std::numeric_limits<Scalar::Real<T>>::min();
        
        template<typename T> constexpr Scalar::Real<T> Max      = std::numeric_limits<Scalar::Real<T>>::max();
        
        template<typename T> constexpr Scalar::Real<T> Infty    = std::numeric_limits<Scalar::Real<T>>::has_infinity ? std::numeric_limits<Scalar::Real<T>>::infinity() : std::numeric_limits<Scalar::Real<T>>::max();
        
        template<typename T> constexpr Scalar::Real<T> Min      = std::numeric_limits<Scalar::Real<T>>::lowest();
        
    } // namespace Scalar
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr T Conj( cref<T> z )
    {
        if constexpr ( Scalar::ComplexQ<T> )
        {
            return std::conj(z) ;
        }
        else
        {
            return z;
        }
    }
    
    
//    template<typename T>
//    TOOLS_FORCE_INLINE constexpr std::complex<T> Conj( const std::complex<T> & z )
//    {
//        return std::conj(z);
//    }
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr Scalar::Real<T> Re( cref<T> z )
    {
        if constexpr ( Scalar::ComplexQ<T> )
        {
            return std::real(z) ;
        }
        else
        {
            return z;
        }
    }
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr Scalar::Real<T> Im( cref<T> z )
    {
        if constexpr ( Scalar::ComplexQ<T> )
        {
            return std::imag(z) ;
        }
        else
        {
            return static_cast<Scalar::Real<T>>(0);
        }
    }
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr Scalar::Real<T> AbsSquared( cref<T> z )
    {
        if constexpr ( Scalar::ComplexQ<T> )
        {
            return Re(z) * Re(z) + Im(z) * Im(z) ;
        }
        else
        {
            return z * z;
        }
    }
    
    template<typename T>
    TOOLS_FORCE_INLINE constexpr Scalar::Real<T> Abs( cref<T> z )
    {
        using std::abs;
        
        if constexpr ( std::is_floating_point_v<T> )
        {
            return std::fabs(z);
        }
        else if constexpr ( UnsignedIntQ<T> )
        {
            return z;
        }
        else
        {
            return static_cast<Scalar::Real<T>>(abs(z));
        }
    }
    
    
    // scalar_cast<T>(x) cast x to the precision of T, but preserves Real/Complex
    template<typename T, typename S>
    TOOLS_FORCE_INLINE constexpr
    typename std::enable_if_t<
        Scalar::ScalarQ<T> && Scalar::RealQ<S>,
        typename Scalar::Real<T>
    >
    scalar_cast( S x )
    {
        return static_cast<typename Scalar::Real<T>>(x);
    }
    
    template<typename T, typename S>
    TOOLS_FORCE_INLINE constexpr
    typename std::enable_if_t<
        Scalar::ScalarQ<T> && Scalar::ComplexQ<S>,
        typename Scalar::Complex<T>
    >
    scalar_cast( cref<S> x )
    {
        return static_cast<typename Scalar::Complex<T>>(x);
    }
    
    // First converts to Real<R> and then compute the reciprocal.
    template<typename S, typename R = S>
    TOOLS_FORCE_INLINE constexpr R Inv( cref<S> a )
    {
        return Scalar::One<R> / scalar_cast<R>(a);
    }
    
    template<typename R, typename S, typename T>
    TOOLS_FORCE_INLINE constexpr R Frac( cref<S> a, cref<T> b )
    {
        return scalar_cast<R>(a) / scalar_cast<R>(b);
    }
    
//    template<typename S, typename T>
//    TOOLS_FORCE_INLINE constexpr decltype(S(1) * T(1)) Frac( cref<S> a, cref<T> b )
//    {
//        using R = decltype(S(1) * T(1));
//        return scalar_cast<R>(a) / scalar_cast<R>(b);
//    }
    
    template<typename R, typename S, typename T>
    TOOLS_FORCE_INLINE constexpr R Percentage( cref<S> a, cref<T> b )
    {
        static_assert(Scalar::RealQ<S>,"");
        static_assert(Scalar::RealQ<T>,"");
        static_assert(Scalar::RealQ<R>,"");
        return R(100) * scalar_cast<R>(a) / scalar_cast<R>(b);
    }
    
//    template<typename S, typename T>
//    TOOLS_FORCE_INLINE constexpr decltype(S(1) * T(1)) Percentage( cref<S> a, cref<T> b )
//    {
//        using R = decltype(S(1) * T(1));
//        
//        static_assert(Scalar::RealQ<S>,"");
//        static_assert(Scalar::RealQ<T>,"");
//        static_assert(Scalar::RealQ<R>,"");
//        
//        return R(100) * scalar_cast<R>(a) / scalar_cast<R>(b);
//    }
    
//    template<typename Scal>
//    inline constexpr bool NaNQ( cref<Scal> x )
//    {
//        // TODO: Does not work with `-ffast-math` option.
//        if constexpr ( Scalar::RealQ<Scal> )
//        {
//            return std::isnan(x);
//        }
//        else if constexpr ( Scalar::ComplexQ<Scal> )
//        {
//            return NaNQ( Re(x) ) || NaNQ( Im(x) );
//        }
//        else
//        {
//            return true;
//        }
//    }
    
    template<typename Scal>
    inline constexpr bool NaNQ( cref<Scal> x )
    {
        TOOLS_MAKE_FP_STRICT();
        
        // Works also with `-ffast-math` option?
        if constexpr ( FloatQ<Scal> && Scalar::RealQ<Scal> )
        {
//            return std::isnan(x) || (ToString(x) == "nan");
            return std::isnan(x);
        }
        else if constexpr ( Scalar::ComplexQ<Scal> )
        {
            return NaNQ( Re(x) ) || NaNQ( Im(x) );
        }
        else
        {
            return true;
        }
    }
    
    //    template<typename Scal>
    //    inline constexpr bool NaNQ( cref<Scal> x )
    //    {
    //        // This is a work-around to detect NaNs under -ffast-math option.
    //        // Not sure whether this is portable.
    //        if constexpr ( Scalar::RealQ<Scal> )
    //        {
    ////            return !(
    ////                (std::numeric_limits<Scal>::lowest() <= x)
    ////                &&
    ////                ( x <=  std::numeric_limits<Scal>::max() )
    ////            );
    //
    //            // This works at least for Apple clang.
    //            return (x < std::numeric_limits<Scal>::lowest());
    //        }
    //        else if constexpr ( Scalar::ComplexQ<Scal> )
    //        {
    //            return NaNQ( Re(x) ) || NaNQ( Im(x) );
    //        }
    //        else
    //        {
    //            return true;
    //        }
    //    }
    
#define ASSERT_REAL(R) static_assert( Scalar::RealQ<R>, "Template parameter " #R " must be a real-valued type." );
    
#define ASSERT_COMPLEX(C) static_assert( Scalar::ComplexQ<C>, "Template parameter " #C " must be a complex-valued type." );
    
    

    namespace Scalar
    {
        template<const Scalar::Flag a_flag, Tools::Op op, typename a_T, typename x_T>
        [[nodiscard]] TOOLS_FORCE_INLINE constexpr bool OpReturnRealQ()
        {
            // This function returns `true` of it can be guaranteed at compile time that `Op<a_flag,op>(a,x)` can be represented by a type `R` satisfying `Scalar::RealQ<R> == true`.
            
            if constexpr ( a_flag == Flag::Zero )
            {
                // The result would be 0, which is real.
                return true;
            }
            
            if constexpr ( RealQ<x_T> && ( (op == Tools::Op::Im) || (op == Tools::Op::ImTrans) ) )
            {
                // The result would be 0, which is real.
                return true;
            }
            
//            constexpr bool a_Real_Q = RealQ<a_T> || ( ComplexQ<a_T> && RealQ<x_T> );
            
            constexpr bool a_Real_Q = RealQ<a_T> || (a_flag == Flag::Plus) || (a_flag == Flag::Minus);
            
            constexpr bool x_Real_Q = RealQ<x_T>|| (op == Tools::Op::Re) || (op == Tools::Op::Im) || (op == Tools::Op::ReTrans) || (op == Tools::Op::ImTrans);
            
            return a_Real_Q && x_Real_Q;
        }

        
        template<
            Tools::Op op, typename x_T,
            typename Return_T =
            std::conditional_t<
                OpReturnRealQ<Flag::Plus,op,x_T,x_T>(),
                Scalar::Real<x_T>,
                x_T
            >
        >
        TOOLS_FORCE_INLINE constexpr Return_T Op( cref<x_T> x )
        {
            if constexpr ( (op == Tools::Op::Id) || (op == Tools::Op::Trans) )
            {
                return x;
            }
            else if constexpr ( (op == Tools::Op::Conj) || (op == Tools::Op::ConjTrans) )
            {
                return Conj(x);
            }
            else if constexpr ( (op == Tools::Op::Re) || (op == Tools::Op::ReTrans) )
            {
                return Re(x);
            }
            else if constexpr ( (op == Tools::Op::Im) || (op == Tools::Op::ImTrans) )
            {
                return Im(x);
            }
        }
        
        
        template<
            Flag a_flag, typename a_T, typename x_T,
            typename Return_T =
            std::conditional_t<
                OpReturnRealQ<a_flag,Tools::Op::Id,a_T,x_T>(),
                decltype( Real<a_T>(0) * Real<x_T>(0) ),
                decltype( a_T(0) * x_T(0) )
            >
        >
        TOOLS_FORCE_INLINE constexpr Return_T Op( cref<a_T> a, cref<x_T> x )
        {
            if constexpr ( a_flag == Flag::Generic )
            {
                return a * x;
            }
            else if constexpr ( a_flag == Flag::Plus )
            {
                return x;
            }
            else if constexpr ( a_flag == Flag::Minus )
            {
                return -x;
            }
            else if constexpr ( a_flag == Flag::Zero )
            {
                return static_cast<Return_T>(0);
            }
        }
        
        
        
//        template<
//            Scalar::Flag a_flag, typename a_T, typename x_T,
//            typename Return_T = decltype( scalar_cast<x_T>(a_T(0)) * x_T(0) )
//        >
//        TOOLS_FORCE_INLINE constexpr Return_T Op( cref<a_T> a, cref<x_T> x )
//        {
//            if constexpr ( a_flag == Scalar::Flag::Generic )
//            {
//                return a * x;
//            }
//            else if constexpr ( a_flag == Scalar::Flag::Plus )
//            {
//                return x;
//            }
//            else if constexpr ( a_flag == Scalar::Flag::Minus )
//            {
//                return -x;
//            }
//            else if constexpr ( a_flag == Flag::Zero )
//            {
//                return static_cast<Return_T>(0);
//            }
//        }
        
        template<
            Flag a_flag, Tools::Op op,
            typename a_T, typename x_T,
            typename Return_T = decltype( Op<a_flag>( a_T(0), Op<op>(x_T(0)) ) )
        >
        TOOLS_FORCE_INLINE constexpr Return_T Op( cref<a_T> a, cref<x_T> x )
        {
            return Op<a_flag,a_T,x_T,Return_T>( a, Op<op>(x) );
        }
        
    } // namespace Scalar
    
    
} // namespace Tools

