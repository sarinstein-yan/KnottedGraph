#pragma once

namespace Tools
{
    // Special version for purely complex types.
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename x_T, typename b_T, typename y_T, typename R
    >
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<
        VectorizableQ<R>
        && SameQ<Scalar::Real<a_T>,R> && SameQ<Scalar::Real<x_T>,R>
        && SameQ<Scalar::Real<b_T>,R> && SameQ<Scalar::Real<y_T>,R>
        ,
        void
    >
    combine_scalars(
        cref<a_T> a, cref<x_T> x,
        // We explicitly allow y and z to overlap.
        cref<b_T> b, const y_T & y,
        Cplx<R> & z
    )
    {
        // This computes z = a * opx(x) + b * opy(y).
        
        // We use references for the arguments in the case that
        // this would require to dereference some illegal pointer
        // upstream.
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
        using V_T = vec_T<2,R>;
            
        V_T & Z = to_vec_T(z);
        
        {
            TOOLS_MAKE_FP_FAST()
            
            Z = Scalar::Op_vec<a_flag,opx>(a,x)
                +
                Scalar::Op_vec<b_flag,opy>(b,y);
        }
    }
    
} // namespace Tools
