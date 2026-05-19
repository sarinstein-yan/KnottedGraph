#pragma once

namespace Tools
{
    template<bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Real Det3D_Naive( cptr<Real> A )
    {
        if constexpr( fmaQ )
        {
            const Real r [3] {
                fma( A[4], A[8], - A[5] * A[7] ),
                fma( A[5], A[6], - A[3] * A[8] ),
                fma( A[3], A[7], - A[4] * A[6] )
            };
            
            return fma( A[2], r[2], fma( A[1], r[1], A[0] * r[0] ) );
        }
        else
        {
            return
                A[0] * (A[4] * A[8] - A[5] * A[7])
                +
                A[1] * (A[5] * A[6] - A[3] * A[8])
                +
                A[2] * (A[3] * A[7] - A[4] * A[6]);
        }
    }
    
    template<bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Real SignedVolume3D_Naive(
        cptr<Real> x, cptr<Real> y, cptr<Real> z, cptr<Real> w
    )
    {
        const Real A [3][3] {
            { y[0] - x[0], z[0] - x[0], w[0] - x[0] },
            { y[1] - x[1], z[1] - x[1], w[1] - x[1] },
            { y[2] - x[2], z[2] - x[2], w[2] - x[2] }
        };
        
        return Det3D_Naive<fmaQ>( &A[0][0] );
    }
    
    template<typename Out_T = int, bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Out_T Oriented3D_Naive(
        cptr<Real> x, cptr<Real> y, cptr<Real> z, cptr<Real> w
    )
    {
        return Sign<Out_T>( SignedVolume3D_Naive<fmaQ>( x, y, z, w ) );
    }
        
} // namespace Tools
