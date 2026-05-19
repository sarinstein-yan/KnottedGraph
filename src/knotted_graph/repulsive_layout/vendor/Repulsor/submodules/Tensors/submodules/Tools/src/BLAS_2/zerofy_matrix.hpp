#pragma once

namespace Tools
{
    template <
        Size_T M, Size_T N, Parallel_T parQ,
        typename X_T
    >
    TOOLS_FORCE_INLINE constexpr void zerofy_matrix(
        mptr<X_T> X, const Size_T ldX,
        const Size_T m, const Size_T n, const Size_T thread_count
    )
    {
        if( (ldX == n) )
        {
            // Consecutive buffer. Do a standard zerofy.
            if constexpr ( (M > VarSize) && (N > VarSize ) )
            {
                zerofy_buffer<M*N,Sequential>( X );
            }
            else
            {
                zerofy_buffer<VarSize,parQ>( X, m * n, thread_count );
            }
        }
        else
        {
            Do<M,parQ,Static>(
                [=]( const Size_T i )
                {
                    zerofy_buffer<N,Sequential>( &X[ldX * i], n );
                },
                m, thread_count
            );
        }
    }
    
    
    // Overload for fixed sizes.
    template <
        Size_T M, Size_T N,
        typename X_T
    >
    TOOLS_FORCE_INLINE constexpr void zerofy_matrix(
        mptr<X_T> X, const Size_T ldX
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        zerofy_matrix<M,N,Sequential>( X, ldX, M, N, Size_T(1) );
    }
    
    
    // Overload for variable sizes, parallel evaluation.
    template <
        typename X_T
    >
    TOOLS_FORCE_INLINE constexpr void zerofy_matrix(
        mptr<X_T> X, const Size_T ldX,
        const Size_T m, const Size_T n, const Size_T thread_count
    )
    {
        zerofy_matrix<VarSize,VarSize,Parallel>( X, ldX, m, n, thread_count );
    }
    
    
    // Overload for variable sizes, sequential evaluation.
    template <
        typename X_T
    >
    TOOLS_FORCE_INLINE constexpr void zerofy_matrix(
        mptr<X_T> X, const Size_T ldX,
        const Size_T m, const Size_T n
    )
    {
        zerofy_matrix<VarSize,VarSize,Sequential>( X, ldX, m, n, Size_T(1) );
    }
    
} // namespace Tools

