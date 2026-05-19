#pragma once

namespace Tools
{
    template<bool fmaQ = true, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Interval<Real,RP> Det2D_IA(
        const Singleton<Real,RP> a, const Singleton<Real,RP> b,
        const Singleton<Real,RP> c, const Singleton<Real,RP> d
    )
    {
        if constexpr ( fmaQ )
        {
            return fma( a, d, - b * c );
        }
        else
        {
            return a * d - b * c;
        }
    }
    
    
    
    template<bool fmaQ = true, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Interval<Real,RP> Det2D_IA( cptr<Singleton<Real,RP>> A )
    {
        return Det2D_IA<fmaQ>( A[0], A[1], A[2], A[3] );
    }
    
    template<bool fmaQ = true, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Interval<Real,RP> Det2D_IA(
        const Interval<Real,RP> a, const Interval<Real,RP> b,
        const Interval<Real,RP> c, const Interval<Real,RP> d
    )
    {
        if constexpr ( fmaQ )
        {
            return fma( a, d, - b * c );
        }
        else
        {
            return a * d - b * c;
        }
    }
    
    template<bool fmaQ = true, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Interval<Real,RP> Det2D_IA( cptr<Interval<Real,RP>> A )
    {
        return Det2D_IA<fmaQ>( A[0], A[1], A[2], A[3] );
    }

    
    template<RoundingPolicy RP, bool diffsafeQ = false, bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Interval<Real,RP> SignedVolume2D_IA( cptr<Real> x, cptr<Real> y, cptr<Real> z )
    {
        using S_T = Singleton<Real,RP>;
        
        if constexpr ( diffsafeQ )
        {
            return Det2D_IA<fmaQ>( S_T(y[0] - x[0]), S_T(z[0] - x[0]), S_T(y[1] - x[1]), S_T(z[1] - x[1]) );
        }
        else
        {
            cptr<S_T> X = reinterpret_cast<cptr<S_T>>(x);
            cptr<S_T> Y = reinterpret_cast<cptr<S_T>>(y);
            cptr<S_T> Z = reinterpret_cast<cptr<S_T>>(z);
            
            return Det2D_IA<fmaQ>( Y[0] - X[0], Z[0] - X[0], Y[1] - X[1], Z[1] - X[1] );
        }
    }
    
    

    
    template<typename Out_T = int, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Out_T DetSign2D_IA(
        const Singleton<Real,RP> a, const Singleton<Real,RP> b,
        const Singleton<Real,RP> c, const Singleton<Real,RP> d
    )
    {
        return DifferenceSign<Out_T>( a * d, b * c );
    }
    
    template<typename Out_T = int, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Out_T DetSign2D_IA( cptr<Singleton<Real,RP>> A )
    {
        return DetSign2D_IA( A[0], A[1], A[2], A[3] );
    }
    
//    template<typename Out_T = int, typename Real>
//    TOOLS_FORCE_INLINE Out_T DetSign2D_IA(
//        const Real a, const Real b,
//        const Real c, const Real d
//    )
//    {
//        using S_T = Singleton<Real,RP>;
//
//        return DifferenceSign<Out_T>( S_T(a) * S_T(d), S_T(b) * S_T(c) );
//    }

//    template<typename Out_T = int, typename Real>
//    TOOLS_FORCE_INLINE Out_T DetSign2D_IA( cptr<Real> A )
//    {
//        return DetSign2D_IA<Out_T>( A[0], A[1], A[2], A[3] );
//    }
    
    template<typename Out_T = int, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Out_T DetSign2D_IA(
        const Interval<Real,RP> a, const Interval<Real,RP> b,
        const Interval<Real,RP> c, const Interval<Real,RP> d
    )
    {
        return DifferenceSign<Out_T>( a * d, b * c );
    }
    
//    template<typename Out_T = int, typename Real>
//    TOOLS_FORCE_INLINE Out_T DetSign2D_IA( cptr<Interval<Real>> A )
//    {
//        return DetSign2D_IA( A[0], A[1], A[2], A[3] );
//    }
    
    template<RoundingPolicy RP, typename Out_T, bool diffsafeQ, typename Real>
    TOOLS_FORCE_INLINE Out_T Oriented2D_IA( cptr<Real> x, cptr<Real> y, cptr<Real> z )
    {
        using S_T = Singleton<Real,RP>;
        
        if constexpr ( diffsafeQ )
        {
            // This calls the Singleton<Real,RP> overload of DetSign2D_IA.
            return DetSign2D_IA<Out_T>( S_T(y[0] - x[0]), S_T(z[0] - x[0]), S_T(y[1] - x[1]), S_T(z[1] - x[1]) );
        }
        else
        {
            
            cptr<S_T> X = reinterpret_cast<cptr<S_T>>(x);
            cptr<S_T> Y = reinterpret_cast<cptr<S_T>>(y);
            cptr<S_T> Z = reinterpret_cast<cptr<S_T>>(z);
            
            // This calls the Interval<Real,RP> overload of DetSign2D_IA.
            return DetSign2D_IA<Out_T>( Y[0] - X[0], Z[0] - X[0], Y[1] - X[1], Z[1] - X[1] );
        }
    }
    
} // namespace Tools

