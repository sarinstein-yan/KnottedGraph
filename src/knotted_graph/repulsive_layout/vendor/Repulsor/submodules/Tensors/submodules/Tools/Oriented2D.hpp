#pragma once

namespace Tools
{

    template<bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Real Det2D_Naive( const Real a, const Real b, const Real c, const Real d )
    {
        if constexpr ( fmaQ )
        {
            return std::fma( a, d, - b * c );
        }
        else
        {
            return a * d - b * c;
        }
    }
    
    template<bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Real Det2D_Naive( cptr<Real> A )
    {
        return Det2D_Naive<fmaQ>( A[0], A[1], A[2], A[3] );
    }

    template<typename Out_T = int, typename Real>
    TOOLS_FORCE_INLINE Out_T DetSign2D_Naive( const Real a, const Real b, const Real c, const Real d )
    {
        return DifferenceSign<Out_T>( a * d, b * c );
    }
    
    template<typename Out_T = int, typename Real>
    TOOLS_FORCE_INLINE Out_T DetSign2D_Naive( cptr<Real> A )
    {
        return DetSign2D_Naive<Out_T>( A[0], A[1], A[2], A[3] );
    }
    
    template<bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Real SignedVolume2D_Naive( cptr<Real> x, cptr<Real> y, cptr<Real> z )
    {
        return Det2D_Naive<fmaQ>( y[0] - x[0], z[0] - x[0], y[1] - x[1], z[1] - x[1] );
    }
    
    template<typename Out_T = int, typename Real>
    TOOLS_FORCE_INLINE Out_T Oriented2D_Naive( cptr<Real> x, cptr<Real> y, cptr<Real> z )
    {
        return DifferenceSign<Out_T>( (y[0] - x[0]) * (z[1] - x[1]), (y[1] - x[1]) * (z[0] - x[0]) );
    }

    
    template<typename Out_T = int, typename Real>
    TOOLS_FORCE_INLINE Out_T DetSign2D_Corrected( const Real a, const Real b, const Real c, const Real d )
    {
        static_assert(FloatQ<Real>,"");
        
        const auto [ad,ade] = TwoMulFMA(a,d);
        const auto [bc,bce] = TwoMulFMA(b,c);

        const Real diff = ad  - bc;
        const Real e    = bce - ade;
            
        return DifferenceSign<Out_T>(diff,e);
    }
    
    template<typename Out_T = int, typename Real>
    TOOLS_FORCE_INLINE Out_T DetSign2D_Corrected( cptr<Real> A )
    {
        return DetSign2D_Corrected<Out_T>( A[0], A[1], A[2], A[3] );
    }

    template<typename Real>
    TOOLS_FORCE_INLINE Real Det2D_Corrected( const Real a, const Real b, const Real c, const Real d )
    {
        static_assert(FloatQ<Real>,"");
        
        const auto [ad,ade] = TwoMulFMA(a,d);
        const auto [bc,bce] = TwoMulFMA(b,c);

        const Real diff = ad  - bc;
        const Real e    = bce - ade;
            
        return diff - e;
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE Real Det2D_Corrected( cptr<Real> A )
    {
        return Det2D_Corrected( A[0], A[1], A[2], A[3] );
    }
    
} // namespace Tools
