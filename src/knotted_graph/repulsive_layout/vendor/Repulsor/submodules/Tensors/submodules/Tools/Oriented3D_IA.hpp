#pragma once

namespace Tools
{
//    template<bool fmaQ = true, typename Real>
//    TOOLS_FORCE_INLINE Interval<Real,RP> Det3D_IA( cptr<Real> A )
//    {
//        return Det3D_IA( reinterpret_cast<cptr<Singleton<Real>>>(A) );
//    }
    
    template<bool fmaQ = true, typename Real, RoundingPolicy RP>
    TOOLS_FORCE_INLINE Interval<Real,RP> Det3D_IA( cptr<Interval<Real,RP>> A )
    {
        if constexpr( fmaQ )
        {
            const Interval<Real,RP> r [3] {
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
    
    template<RoundingPolicy RP, bool diffsafeQ = false, bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Interval<Real,RP> SignedVolume3D_IA(
        cptr<Real> x, cptr<Real> y, cptr<Real> z, cptr<Real> w
    )
    {
        using S_T = Singleton<Real,RP>;
        using I_T = Interval <Real,RP>;

        if constexpr ( diffsafeQ )
        {
            const S_T A [3][3] {
                { S_T(y[0] - x[0]), S_T(z[0] - x[0]), S_T(w[0] - x[0]) },
                { S_T(y[1] - x[1]), S_T(z[1] - x[1]), S_T(w[1] - x[1]) },
                { S_T(y[2] - x[2]), S_T(z[2] - x[2]), S_T(w[2] - x[2]) }
            };
            
            return Det3D_IA<fmaQ>( &A[0][0] );
        }
        else
        {
            cptr<S_T> X = reinterpret_cast<cptr<S_T>>(x);
            cptr<S_T> Y = reinterpret_cast<cptr<S_T>>(y);
            cptr<S_T> Z = reinterpret_cast<cptr<S_T>>(z);
            cptr<S_T> W = reinterpret_cast<cptr<S_T>>(w);

            const I_T A [3][3] {
                { Y[0] - X[0], Z[0] - X[0], W[0] - X[0] },
                { Y[1] - X[1], Z[1] - X[1], W[1] - X[1] },
                { Y[2] - X[2], Z[2] - X[2], W[2] - X[2] }
            };
            
            return Det3D_IA<fmaQ>( &A[0][0] );
        }
    }
    
    
    template<RoundingPolicy RP, typename Out_T = int, bool diffsafeQ = false, bool fmaQ = true, typename Real>
    TOOLS_FORCE_INLINE Out_T Oriented3D_IA(
        cptr<Real> x, cptr<Real> y, cptr<Real> z, cptr<Real> w
    )
    {
        return Sign<Out_T>( SignedVolume3D_IA<RP,diffsafeQ,fmaQ>(x,y,z,w) );
    }
    
} // namespace Tools

