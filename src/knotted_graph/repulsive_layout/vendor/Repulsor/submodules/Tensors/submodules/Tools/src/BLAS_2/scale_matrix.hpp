#pragma once

namespace Tools
{
    /*!
     * @brief Computes `Y = beta * Y`, where `X` is a matrix of size `m` x `n`.
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
     * @param beta A scalar.
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
        typename b_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void scale_matrix(
        cref<b_T> beta, mptr<Y_T> Y, const Size_T ldY,
        const Size_T m, const Size_T n, const Size_T thread_count
    )
    {
        const auto beta_ = scalar_cast<Y_T>(beta);
        
        if( (ldY == n) )
        {
            // Consecutive buffer. Do a standard scale_buffer.
            if constexpr ( (M > VarSize) && (N > VarSize ) )
            {
                scale_buffer<M * N,Sequential>( beta_, Y );
            }
            else
            {
                scale_buffer<VarSize,Parallel>( beta_, Y, m * n, thread_count );
            }
        }
        else
        {
            Do<M,parQ,Static>(
                [=]( const Size_T i )
                {
                    scale_buffer<N,Sequential>( beta_, &Y[ldY * i], n );
                },
                m, thread_count
            );
        }
    }
    
    
    // Overload for fixed sizes.
    template <
        Size_T M, Size_T N,
        typename b_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void scale_matrix(
        cref<b_T> beta, mptr<Y_T> Y, const Size_T ldY
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        scale_matrix<M,N,Sequential>( beta, Y, ldY, M, N, Size_T(1) );
    }
    
    
    // Overload for variable sizes, parallel evaluation.
    template <
        typename b_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void scale_matrix(
        cref<b_T> beta, mptr<Y_T> Y, const Size_T ldY,
        const Size_T m, const Size_T n, const Size_T thread_count
    )
    {
        scale_matrix<VarSize,VarSize,Parallel>( beta, Y, ldY, m, n, thread_count );
    }
    
    // Overload for variable sizes, sequential evaluation.
    template <
        typename b_T, typename Y_T
    >
    TOOLS_FORCE_INLINE constexpr void scale_matrix(
        cref<b_T> beta, mptr<Y_T> Y, const Size_T ldY,
        const Size_T m, const Size_T n
    )
    {
        scale_matrix<VarSize,VarSize,Sequential>( beta, Y, ldY, m, n, Size_T(1) );
    }
    
} // namespace Tools


