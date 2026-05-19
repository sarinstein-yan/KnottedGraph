#pragma once

namespace Tools
{
    /*!
     * @brief Computes `Y = X`, where `X` and `Y` are matrices of size `m` x `n`.
     *
     * @tparam M Compile-time knowledge of the number of rows.
     *   `M > 0` asks to use fixed-size loops that can be optimized by the compiler.
     *   `M = 0` uses variable the run-time row count specified by argument `m`.
     *
     * @tparam N Compile-time knowledge of the number of columns.
     *   `N > 0` : Use fixed-size loops that can be optimized by the compiler.
     *   `N = 0` : Use variable the run-time column count specified by argument `n`.
     *
     * @tparam parQ Whether to parallelize or not.
     *
     * @param X A matrix of size `m` x `n`.
     *
     * @param ldX Leading dimension of `X`.
     *
     * @param Y A matrix of size `m` x `n`.
     *
     * @param ldY Leading dimension of `Y`.
     *
     * @param m Number of rows.
     *
     * @param n Number of columns.
     *
     * @param thread_count Number of threads to parallelize over.
     */
    
    template <
        Size_T M, Size_T N, Parallel_T parQ,
        typename X_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void copy_matrix(
        cptr<X_T> X, const Size_T ldX,
        mptr<Y_T> Y, const Size_T ldY,
        const Size_T m, const Size_T n, const Size_T thread_count
    )
    {
        if( (ldX == n) && (ldY == n) )
        {
            // Consecutive buffer. Do a standard copy.
            
            if constexpr ( (M > VarSize) && (N > VarSize ) )
            {
                copy_buffer<M*N>( X, Y );
            }
            else
            {
                copy_buffer<VarSize,Parallel>( X, Y, m * n, thread_count);
            }
        }
        else
        {
            Do<M,parQ,Static>(
                [=]( const Size_T i )
                {
                    copy_buffer<N,Sequential>( &X[ldX * i], &Y[ldY * i], n );
                },
                m, thread_count
            );
        }
    }
    
    // Overload for fixed sizes.
    template <
        Size_T M, Size_T N,
        typename X_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void copy_matrix(
        cptr<X_T> X, const Size_T ldX,
        mptr<Y_T> Y, const Size_T ldY
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        copy_matrix<M,N,Sequential>( X, ldX, Y, ldY, M, N, Size_T(1) );
    }
    
    
    // Overload for variable sizes, parallel evaluation.
    template <
        typename X_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void copy_matrix(
        cptr<X_T> X, const Size_T ldX,
        mptr<Y_T> Y, const Size_T ldY,
        const Size_T m, const Size_T n, const Size_T thread_count
    )
    {
        copy_matrix<VarSize,VarSize,Parallel>( X, ldX, Y, ldY, m, n, thread_count );
    }
    
    
    // Overload for variable sizes, sequential evaluation.
    template <
        typename X_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void copy_matrix(
        cptr<X_T> X, const Size_T ldX,
        mptr<Y_T> Y, const Size_T ldY,
        const Size_T m, const Size_T n
    )
    {
        copy_matrix<VarSize,VarSize,Parallel>( X, ldX, Y, ldY, m, n, Size_T(1) );
    }
    
} // namespace Tools
