#pragma once

namespace Tools
{
    template<typename Real>
    TOOLS_FORCE_INLINE std::pair<Real,Real> Det2D_Kahan_DiffPair(
        const Real a, const Real b,
        const Real c, const Real d
    )
    {
        // https://www.johndcook.com/blog/2020/05/31/kahan-determinant/
        
        // Computes a * d - b * c with less rounding error.
        
        
//        const Real w = b * c;
//        const Real e = std::fma(-b,c, w);
//        const Real f = std::fma( a,d,-w);
//
        // Precise determinat would be e + f;
        
        TOOLS_MAKE_FP_STRICT()
        
        // This should use one negation less than the above.
        const Real w = b * c;
        const Real e = std::fma( -b, c, w );
        const Real f = std::fma( -a, d, w );
        
        // The precise determinant is e - f.
        return {e,f};
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE std::pair<Real,Real> Dot2D_Kahan_DiffPair(
        const Real u_0, const Real u_1,
        const Real v_0, const Real v_1
    )
    {
        // The precise dot product is e - f.
        return Det2D_Kahan_DiffPair( u_0, u_1, -v_1, v_0 );
    }
    
    
    
    template<typename Real>
    TOOLS_FORCE_INLINE Real Det2D_Kahan(
        const Real a, const Real b,
        const Real c, const Real d
    )
    {
        auto [e,f] = Det2D_Kahan_DiffPair(a,b,c,d);
        
        return e - f;
    }

    template<typename Out_T = FastInt8, typename Real>
    TOOLS_FORCE_INLINE Out_T DetSign2D_Kahan(
        const Real a, const Real b,
        const Real c, const Real d
    )
    {
        auto [e,f] = Det2D_Kahan_DiffPair(a,b,c,d);
        
        return DifferenceSign<Out_T>(e,f);
    }

    template<typename Real>
    TOOLS_FORCE_INLINE Real Det2D_Kahan( cptr<Real> A )
    {
        return Det2D_Kahan( A[0], A[1], A[2], A[3] );
    }
    
    
    template<typename Real>
    TOOLS_FORCE_INLINE Real Dot2D_Kahan(
        const Real u_0, const Real u_1,
        const Real v_0, const Real v_1
    )
    {
        return Det2D_Kahan( u_0, u_1, -v_1, v_0 );
    }
    
    template<typename Out_T = FastInt8, typename Real>
    TOOLS_FORCE_INLINE Out_T DotSign2D_Kahan(
        const Real u_0, const Real u_1,
        const Real v_0, const Real v_1
    )
    {
        return DetSign2D_Kahan<Out_T>( u_0, u_1, -v_1, v_0 );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE Real Dot2D_Kahan( cptr<Real> u, cptr<Real> v )
    {
        return Det2D_Kahan( u[0], u[1], -v[1], v[0] );
    }
    
    template<typename Out_T = FastInt8, typename Real>
    TOOLS_FORCE_INLINE Real DotSign2D_Kahan( cptr<Real> u, cptr<Real> v )
    {
        return DetSign2D_Kahan<Out_T>( u[0], u[1], -v[1], v[0] );
    }
    
    template<typename Out_T = FastInt8, typename Real>
    TOOLS_FORCE_INLINE Out_T DetSign2D_Kahan( cptr<Real> A )
    {
        return DetSign2D_Kahan<Out_T>( A[0], A[1], A[2], A[3] );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE Real SignedVolume2D_Kahan( cptr<Real> x, cptr<Real> y, cptr<Real> z )
    {
        return Det2D_Kahan( y[0]-x[0], z[0]-x[0], y[1]-x[1], z[1]-x[1] );
    }
    
    
    template<typename Out_T = FastInt8, typename Real>
    TOOLS_FORCE_INLINE Out_T Oriented2D_Kahan( cptr<Real> x, cptr<Real> y, cptr<Real> z )
    {
        return DetSign2D_Kahan<Out_T>( y[0]-x[0], z[0]-x[0], y[1]-x[1], z[1]-x[1] );
    }
    
} // namespace Tools
