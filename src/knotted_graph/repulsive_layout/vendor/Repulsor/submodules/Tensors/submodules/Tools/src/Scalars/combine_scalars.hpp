#pragma once

namespace Tools
{
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename x_T, typename b_T, typename y_T
    >
    constexpr TOOLS_FORCE_INLINE
    void combine_scalars(
        cref<a_T> a, cref<x_T> x, cref<b_T> b, mref<y_T> y
    )
    {
        // Computes y = a * opx( x ) + b * opy( y ).
        
        // We use references for the arguments in the case that
        // this would require to dereference some illegal pointer
        // upstream.
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
        static_assert( 
            Scalar::ComplexQ<y_T>
            ||
            Scalar::RealQ<decltype( Scalar::Op<a_flag,opx>( a_T(0), x_T(0) ) )>
            ,
            "Fourth argument is real, but some of the other arguments are complex."
        );
        
        // We refrain from automagically casting `a` and `b` to the right precision because this is better done once before any loop that calls `combine_scalars`. Hence, we prefer a compile error here.
        
        static_assert(
            Scalar::Prec<a_T> == Scalar::Prec<x_T>,
            "Precisions of first and second argument do not coincide."
        );
        
        static_assert(
            Scalar::Prec<b_T> == Scalar::Prec<y_T>,
            "Precisions of third and fourth argument do not coincide."
        );

        auto a_opx_x = Scalar::Op<a_flag,opx>(a,x);
        auto b_opy_y = Scalar::Op<b_flag,opy>(b,y);
        
        y = scalar_cast<decltype(b_opy_y)>(a_opx_x) + b_opy_y;
    }
    
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename x_T, typename b_T, typename y_T, typename z_T
    >
    constexpr TOOLS_FORCE_INLINE
    void combine_scalars(
        cref<a_T> a, cref<x_T> x, cref<b_T> b, cref<y_T> y, mref<z_T> z
    )
    {
        // This computes z = a * opx( x ) + b * opy( y ).
        
        // We use references for the arguments in the case that
        // this would require to dereference some illegal pointer
        // upstream.
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
        static_assert(
            Scalar::ComplexQ<z_T>
            ||
            Scalar::RealQ<decltype( Scalar::Op<a_flag,opx>( a_T(0), x_T(0) ) + Scalar::Op<b_flag,opy>( b_T(0), y_T(0) ) )>
            ,
            "Fourth argument is real, but some of the other arguments are complex."
        );
        
        // We refrain from automagically casting `a` and `b` to the right precision because this is better done once before any loop that calls `combine_scalars`. Hence, we prefer a compile error here.
        
        static_assert(
            Scalar::Prec<a_T> == Scalar::Prec<z_T>,
            "Precisions of first and last argument do not coincide."
        );
        
        static_assert(
            Scalar::Prec<b_T> == Scalar::Prec<z_T>,
            "Precisions of third and last argument do not coincide."
        );
        
        auto a_opx_x = Scalar::Op<a_flag,opx>(a,x);
        auto b_opy_y = Scalar::Op<b_flag,opy>(b,y);
        
        z = scalar_cast<z_T>(a_opx_x) + scalar_cast<z_T>(b_opy_y);
    }
    
} // namespace Tools
