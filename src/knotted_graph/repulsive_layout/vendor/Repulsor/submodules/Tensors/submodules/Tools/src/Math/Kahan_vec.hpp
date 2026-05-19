#pragma once

namespace Tools
{
    template<Size_T n, typename Real>
    TOOLS_FORCE_INLINE
    std::enable_if_t<VectorizableQ<Real>,vec_T<n,Real>>
    Det2D_Kahan(
        cref<vec_T<n,Real>> a, cref<vec_T<n,Real>> b,
        cref<vec_T<n,Real>> c, cref<vec_T<n,Real>> d
    )
    {
        using V_T = vec_T<n,Real>;
        
        const V_T w = b * c;
        const V_T e = __builtin_elementwise_fma( -b, c, w );
        const V_T f = __builtin_elementwise_fma( -a, d, w );
        
        return e - f;
    }
    
    template<Size_T n, typename Real >
    TOOLS_FORCE_INLINE
    std::enable_if_t<
        VectorizableQ<Real>,
        decltype(vec_T<n,Real>{} > vec_T<n,Real>{})
    >
    DetSign2D_Kahan(
        cref<vec_T<n,Real>> a, cref<vec_T<n,Real>> b,
        cref<vec_T<n,Real>> c, cref<vec_T<n,Real>> d
    )
    {
        const vec_T<n,Real> w = b * c;
        const vec_T<n,Real> e = __builtin_elementwise_fma( -b, c, w );
        const vec_T<n,Real> f = __builtin_elementwise_fma( -a, d, w );
        
        return DifferenceSign(e,f);
    }
    
    
    template<Size_T n, typename Real>
    TOOLS_FORCE_INLINE
    std::enable_if_t<
        VectorizableQ<Real>,
        decltype(vec_T<n,Real>{} > vec_T<n,Real>{} )
    >
    Oriented2D_Kahan(
        cref<vec_T<n,Real>> x_0, cref<vec_T<n,Real>> x_1,
        cref<vec_T<n,Real>> y_0, cref<vec_T<n,Real>> y_1,
        cref<vec_T<n,Real>> z_0, cref<vec_T<n,Real>> z_1
    )
    {
        return DetSign2D_Kahan( y_0 - x_0, z_0 - x_0, y_1 - x_1, z_1 - x_1 );
    }
    
    template<typename Real>
    TOOLS_FORCE_INLINE
    std::enable_if_t<VectorizableQ<Real>,bool>
    LineSegmentsIntersectQ_Kahan(
        cref<vec_T<2,Real>> x_0, cref<vec_T<2,Real>> x_1,
        cref<vec_T<2,Real>> y_0, cref<vec_T<2,Real>> y_1
    )
    {
        // Attempt to vectorize the computation of the four determinants.
        // On Apple Silicon this does not seem to be faster, though.
        
        // x_0, x_1, y_0, y_1 are supposed to be points in the plane.
        
        //  x_1    w_1    y_1
        //      X------>X
        //      ^^     ^^
        //      | \q p/ |
        //      |  \ /  |
        //    u |   /   | v
        //      |  / \  |
        //      | /   \ |
        //      |/     \|
        //      X------>X
        //  x_0    w_0    y_0
        
        using V2_T = vec_T<2,Real>;
        using V4_T = vec_T<4,Real>;

        V2_T u = x_1 - x_0;
        V2_T v = y_1 - y_0;
        V2_T p = y_1 - x_0;
        V2_T q = x_1 - y_0;
        
        V4_T A { v[0], p[0], u[0], q[0] };
        V4_T B { v[1], p[1], u[1], q[1] };
        V4_T C { q[0], v[0], p[0], u[0] };
        V4_T D { q[1], v[1], p[1], u[1] };
        
        auto signs = DetSign2D_Kahan(A,B,C,D);

        return (
            ( signs[0] != 0        )
            &&
            ( signs[1] != 0        )
            &&
            ( signs[0] != signs[1] )
            &&
            ( signs[0] == signs[3] )
            &&
            ( signs[1] == signs[2] )
        );
    }
    
    
//    template<typename Real>
//    TOOLS_FORCE_INLINE bool ConvexPlanarQuadrilateralQ_Kahan(
//        cptr<Real> x_0, cptr<Real> x_1, cptr<Real> x_2, cptr<Real> x_3
//    )
//    {
//        // TODO: One can probably boil down the number of subtractions by half here.
//
//        using T = signed char;
//
//        const T signs [4] {
//            Oriented2D_Kahan<T>( x_1, x_2, x_3 ),
//            Oriented2D_Kahan<T>( x_2, x_3, x_0 ),
//            Oriented2D_Kahan<T>( x_3, x_0, x_1 ),
//            Oriented2D_Kahan<T>( x_0, x_1, x_2 )
//        };
//
//        return (
//            ( signs[0] != T(0)     )
//            &&
//            ( signs[0] == signs[1] )
//            &&
//            ( signs[1] == signs[2] )
//            &&
//            ( signs[2] == signs[3] )
//        );
//    }
    
} // namespace Tools
