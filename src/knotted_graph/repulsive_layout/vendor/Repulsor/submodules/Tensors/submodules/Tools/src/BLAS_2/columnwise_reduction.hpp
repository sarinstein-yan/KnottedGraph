#pragma once

namespace Tools
{
    template<
        Size_T M, Size_T N,
        typename X_T, typename y_T, typename R_y_T
    >
    TOOLS_FORCE_INLINE constexpr void columnwise_reduction_vec(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy, R_y_T && y_reducer
    )
    {
        TOOLS_MAKE_FP_FAST();
        
        static_assert(M > VarSize,"");
        static_assert(N > VarSize,"");

        using V_T = vec_T<N,X_T>;
        
        V_T x_vec;
        V_T y_vec;
        
        
        copy_buffer<N>( X, get_ptr(y_vec) );
        
        for( Size_T i = 1; i < M; ++i )
        {
            copy_buffer<N>( &X[ldX * i], get_ptr(x_vec));
            y_vec = y_reducer( y_vec, x_vec );
        }
        
        copy_matrix<N,1>( get_ptr(y_vec), Size_T(1), y, ldy );
    }
    
    template<
        Size_T M, Size_T N,
        typename X_T, typename y_T, typename R_y_T, typename z_T, typename R_z_T
    >
    TOOLS_FORCE_INLINE constexpr void columnwise_reduction_vec(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy, R_y_T && y_reducer,
        mptr<z_T> z,  const Size_T ldz, R_z_T && z_reducer
    )
    {
//        if constexpr ( M <= VarSize ) { return; }

        static_assert(M > VarSize,"");
        static_assert(N > VarSize,"");
        
        using V_T = vec_T<N,X_T>;
        
        V_T x_vec;
        V_T y_vec;
        V_T z_vec;
        
        
        copy_buffer<N>( X, get_ptr(x_vec) );
        
        y_vec = x_vec;
        z_vec = x_vec;
        
        for( Size_T i = 1; i < M; ++i )
        {
            copy_buffer<N>( &X[ldX * i], get_ptr(x_vec) );
            
            y_vec = y_reducer( y_vec, x_vec );
            z_vec = z_reducer( z_vec, x_vec );
        }
        
        copy_matrix<N,1>( get_ptr(y_vec), Size_T(1), y, ldy );
        copy_matrix<N,1>( get_ptr(z_vec), Size_T(1), z, ldz );
    }
    
    template<
        Size_T M, Size_T N,
        typename X_T, typename y_T, typename R_y_T
    >
    TOOLS_FORCE_INLINE constexpr void columnwise_reduction(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy, R_y_T && y_reducer
    )
    {
        TOOLS_MAKE_FP_FAST();
        
//        if constexpr ( M <= VarSize ) { return; }

        static_assert(M > VarSize,"");
        static_assert(N > VarSize,"");
        
        X_T y_buf [N];
        
        copy_buffer<N>( X, &y_buf[0] );
        
        for( Size_T i = 1; i < M; ++i )
        {
            for( Size_T j = 0; j < N; ++j )
            {
                y_buf[j] = y_reducer( y_buf[j], X[ldX * i + j] );
            }
        }
        
        copy_matrix<N,1>( &y_buf[0], Size_T(1), y, ldy );
    }
    
    template<
        Size_T M, Size_T N, typename X_T,
        typename y_T, typename R_y_T, typename z_T, typename R_z_T
    >
    TOOLS_FORCE_INLINE constexpr void columnwise_reduction(
        cptr<X_T> X,  const Size_T ldX,
        mptr<y_T> y,  const Size_T ldy, R_y_T && y_reducer,
        mptr<z_T> z,  const Size_T ldz, R_z_T && z_reducer
    )
    {
        TOOLS_MAKE_FP_FAST();
        
//        if constexpr ( M <= VarSize ) { return; }
        
        static_assert(M > VarSize,"");
        static_assert(N > VarSize,"");

        X_T y_buf [N];
        X_T z_buf [N];
        
        copy_buffer<N>( X, &y_buf[0] );
        copy_buffer<N>( X, &z_buf[0] );
        
        for( Size_T i = 1; i < M; ++i )
        {
            for( Size_T j = 0; j < N; ++j )
            {
                y_buf[j] = y_reducer( y_buf[j], X[ldX * i + j] );
                z_buf[j] = z_reducer( z_buf[j], X[ldX * i + j] );
            }
        }
        
        copy_matrix<N,1>( &y_buf[0], Size_T(1), y, ldy );
        copy_matrix<N,1>( &z_buf[0], Size_T(1), z, ldz );
    }
    
} // namespace Tools
