#pragma once

namespace Tools
{
    template < Size_T M, Size_T N, typename X_T, typename y_T>
    TOOLS_FORCE_INLINE constexpr void columnwise_min(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy
    )
    {
        if constexpr ( VectorizableQ<X_T> )
        {
            using V_T = vec_T<N,X_T>;
            
            columnwise_reduction_vec<M,N>(
                X, ldX,
                y, ldy, [](cref<V_T> a,cref<V_T> b){ return __builtin_elementwise_min(a,b); }
            );
        }
        else
        {
            columnwise_reduction<M,N>(
                X, ldX,
                y, ldy, []( cref<X_T> a, cref<X_T> b ) { return Min(a,b); }
            );
        }
    }
    
    template < Size_T M, Size_T N, typename X_T, typename y_T>
    TOOLS_FORCE_INLINE constexpr void columnwise_max(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy
    )
    {
        if constexpr ( VectorizableQ<X_T> )
        {
            using V_T = vec_T<N,X_T>;
            
            columnwise_reduction_vec<M,N>(
                X, ldX,
                y, ldy, [](cref<V_T> a,cref<V_T> b){ return __builtin_elementwise_max(a,b);}
            );
        }
        else
        {
            columnwise_reduction<M,N>(
                X, ldX,
                y, ldy, [](cref<X_T> a,cref<X_T> b){ return Max(a,b); }
            );
        }
    }
    
    template<Size_T M, Size_T N, typename X_T, typename y_T, typename z_T>
    TOOLS_FORCE_INLINE constexpr void columnwise_minmax(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy,
        mptr<z_T> z,  const Size_T ldz
    )
    {
//        if constexpr ( VectorizableQ<X_T> )
//        {
//            using V_T = vec_T<N,X_T>;
//            
//            columnwise_reduction_vec<M,N>(
//                X, ldX,
//                y, ldy,
//                [](cref<V_T> a,cref<V_T> b){ return __builtin_elementwise_min(a,b); },
//                z, ldz,
//                [](cref<V_T> a,cref<V_T> b){ return __builtin_elementwise_max(a,b); }
//            );
//        }
//        else
        {
            columnwise_reduction<M,N>(
                X, ldX,
                y, ldy, [](cref<X_T> a,cref<X_T> b){ return Min(a,b); },
                z, ldz, [](cref<X_T> a,cref<X_T> b){ return Max(a,b); }
            );
        }
    }
    
} // namespace Tools
