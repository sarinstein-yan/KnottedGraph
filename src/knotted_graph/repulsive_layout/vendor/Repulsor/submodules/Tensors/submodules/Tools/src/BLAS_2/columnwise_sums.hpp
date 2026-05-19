#pragma once

namespace Tools
{
    template < Size_T M, Size_T N, typename X_T, typename y_T>
    TOOLS_FORCE_INLINE constexpr void columnwise_sums(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy
    )
    {
        if constexpr ( VectorizableQ<X_T> )
        {
            using V_T = vec_T<N,X_T>;
            
            columnwise_reduction_vec<M,N>(
                X, ldX, y, ldy, []( cref<V_T> a, cref<V_T> b ) { return a + b; }
            );
        }
        else
        {
            columnwise_reduction<M,N>(
                X, ldX, y, ldy, []( cref<X_T> a, cref<X_T> b ) { return a + b; }
            );
        }
    }
        
} // namespace Tools
