#pragma once

namespace Tools
{
    /*!
     * @brief Computes X = a * opx(X), where `X`
     * is a matrix of size `m` x `n` and `a` is a scalar.
     *
     * @tparam a_flag Specify code optimizations for the scalar `a`.
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
     * @tparam opx Apply a scalar transformation to `X`. Possible values:
     *   `opx = Op::In`   : Apply the identity to the elements of `X`.
     *   `opx = Op::Conj` : Apply `std::Conj` to the elements of `X`.
     *
     *
     * @param a A scalar.
     *
     * @param X A matrix of size `m` x `n`.
     *
     * @param ldX Leading dimension of `X`.
     *
     * @param m Number of rows.
     *
     * @param n Number of columns.
     *
     * @param thread_count Number of threads to parallelize over.
     */
    
    template<
        Scalar::Flag a_flag,
        Size_T M, Size_T N, Parallel_T parQ,
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX,
        const Int m, const Int n, const Int thread_count
    )
    {
        // It is safer to enforce the implicit assumptions explicitly here.
        const Int m_ = (M > VarSize ? static_cast<Int>(M) : m);
        const Int n_ = (N > VarSize ? static_cast<Int>(N) : n);
        
        // PROBLEM: If x == nullptr, which is allowed when a_flag = Zero,
        // then we should better prevent dereferencing x.
        // Otherwise, with optimizations turned off, it could happen that nullptr
        // or some increment of it is dereferenced.
        if constexpr ( (a_flag == Scalar::Flag::Plus) && ( opx == Op::Id ) )
        {
            // Do nothing;
        }
        else
        {
            
            if( ldX == n_ )
            {
                modify_buffer<a_flag,M*N,parQ,opx>(
                    a, X, m_ * n_, thread_count
                );
            }
            else
            {
                Do<M,parQ,Static>(
                    [=]( const Int i )
                    {
                        modify_buffer<a_flag,N,Sequential,opx>( a, &X[ldX * i], n_ );
                    },
                    m_, thread_count
                );
            }
        }
    }
    
    /*!
     * @brief Overload for fixed sizes.
     */
    
    template<
        Scalar::Flag a_flag,
        Size_T M, Size_T N,
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        modify_matrix<a_flag,M,N,Sequential,opx>(
            a, X, ldX,
            M, N, Int(1)
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, parallel evaluation.
     */
    
    template<
        Scalar::Flag a_flag,
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX,
        const Int m, const Int n, const Int thread_count
    )
    {
        modify_matrix<a_flag,VarSize,VarSize,Parallel,opx>(
            a, X, ldX,
            m, n, thread_count
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, sequential evaluation.
     */

    template<
        Scalar::Flag a_flag,
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX,
        const Int m, const Int n
    )
    {
        modify_matrix<a_flag,VarSize,VarSize,Sequential,opx>(
            a, X, ldX,
            m, n, Int(1)
         );
    }
    
    
    
    
    
    
    
    
    /*!
     * @brief Computes `X = a * opx(X)`, where `X` is a matrix of size `m` x `n` and `a` is a scalar. Automatic detection of `a_flag` and `b_flag` at runtime.
     *
     * to different precompiled branches.
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
     * @tparam opx Apply a scalar transformation to `X`. Possible values:
     *   `opx = Op::In`   : Apply the identity to the elements of `X`.
     *   `opx = Op::Conj` : Apply `std::Conj` to the elements of `X`.
     *
     *
     * @param a A scalar.
     *
     * @param X A matrix of size `m` x `n`.
     *
     * @param ldX Leading dimension of `X`.
     *
     * @param m Number of rows.
     *
     * @param n Number of columns.
     *
     * @param thread_count Number of threads to parallelize over.
     */
    
    template<
        Size_T M, Size_T N, Parallel_T parQ,
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix_auto(
        cref<a_T> a, mptr<X_T> X, const Int ldX,
        const Int m, const Int n, const Int thread_count
    )
    {
        using F_T = Scalar::Flag;
        
        auto job = [=]<F_T a_flag>()
        {
            modify_matrix<a_flag,M,N,parQ,opx>(
                a, X, ldX,
                m, n, thread_count
             );
        };
        
        if( a == a_T(1) )
        {
            job.template operator()<F_T::Plus>();
        }
        else if ( a == a_T(0) )
        {
            job.template operator()<F_T::Zero>();
        }
        else if ( a == a_T(-1) )
        {
            job.template operator()<F_T::Minus>();
        }
        else
        {
            job.template operator()<F_T::Generic>();
        }
    }
    
    /*!
     * @brief Overload for fixed sizes. Automatic detection of `a_flag` at runtime.
     */
    
    template<
        Size_T M, Size_T N,
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        modify_matrix_auto<M,N,Sequential,opx>(
            a, X, ldX,
            M, N, Int(1)
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, parallel evaluation. Automatic detection of `a_flag` at runtime.
     */
    template<
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX,
        const Int m, const Int n, const Int thread_count
    )
    {
        modify_matrix_auto<VarSize,VarSize,Parallel,opx>(
            a, X, ldX,
            m, n, thread_count
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, sequential evaluation. Automatic detection of `a_flag` at runtime.
     */
    
    template<
        Op opx = Op::Id,
        typename a_T, typename X_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void modify_matrix(
        cref<a_T> a, mptr<X_T> X, const Int ldX,
        const Int m, const Int n
    )
    {
        modify_matrix_auto<VarSize,VarSize,Sequential,opx>(
            a, X, ldX,
            m, n, Int(1)
         );
    }
    
} // namespace Tools

