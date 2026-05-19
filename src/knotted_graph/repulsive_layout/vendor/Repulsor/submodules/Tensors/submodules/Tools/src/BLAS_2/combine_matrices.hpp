#pragma once

namespace Tools
{
    /*!
     * @brief Computes Y = a * opx(X) + b * opy(Y), where `X` and `Y`
     * are matrices of size `m` x `n` and `a` and `b` are scalars.
     *
     * @tparam a_flag Specify code optimizations for the scalar `a`.
     *
     * @tparam b_flag Specify code optimizations for the scalar `b`.
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
     * @tparam opy Apply a scalar transformation to `Y`. Possible values:
     *   `opy = Op::In`   : Apply the idenity to the elements of `Y`.
     *   `opy = Op::Conj` : Apply `std::Conj` to the elements of `Y`.
     *
     * @param a A scalar.
     *
     * @param X A matrix of size `m` x `n`.
     *
     * @param ldX Leading dimension of `X`.
     *
     * @param b A scalar.
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
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T M, Size_T N, Parallel_T parQ,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY,
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
        if constexpr ( a_flag == Scalar::Flag::Zero )
        {
            // PROBLEM: If y == nullptr, which is allowed when a_flag = One and
            // b_flag = Zero, then we should better prevent dereferencing x.
            // Otherwise, with optimizations turned off, it could happen that nullptr
            // or some increment of is dereferenced.
            if constexpr ( (b_flag == Scalar::Flag::Plus) && ( opy == Op::Id ) )
            {
                // Do nothing;
            }
            else
            {
                modify_matrix<b_flag,VarSize,N,parQ,opy>(
                    b, Y, ldY, m_, n_, thread_count
                );
            }
        }
        else
        {
            if( (ldX == n_) && (ldY == n_) )
            {
                combine_buffers<a_flag,b_flag,M*N,parQ,opx,opy>(
                    a, X, b, Y, m_ * n_, thread_count
                );
            }
            else
            {
                Do<M,parQ,Static>(
                    [=]( const Int i )
                    {
                        combine_buffers<a_flag,b_flag,N,Sequential,opx,opy>(
                            a, &X[ldX * i], b, &Y[ldY * i], n_, Int(1)
                        );
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
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T M, Size_T N,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        combine_matrices<a_flag,b_flag,M,N,Sequential,opx,opy>(
            a, X, ldX,
            b, Y, ldY,
            M, N, Int(1)
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, parallel evaluation.
     */
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY,
        const Int m, const Int n, const Int thread_count
    )
    {
        combine_matrices<a_flag,b_flag,VarSize,VarSize,Parallel,opx,opy>(
            a, X, ldX,
            b, Y, ldY,
            m, n, thread_count
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, sequential evaluation.
     */

    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY,
        const Int m, const Int n
    )
    {
        combine_matrices<a_flag,b_flag,VarSize,VarSize,Sequential,opx,opy>(
            a, X, ldX,
            b, Y, ldY,
            m, n, Int(1)
         );
    }
    
    
    
    
    
    
    
    
    /*!
     * @brief Computes `Y = a * opx(X) + b * opy(Y)`, where `X` and `Y` are matrices of size `m` x `n` and `a` and `b` are scalars. Automatic detection of `a_flag` and `b_flag` at runtime.
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
     * @tparam opy Apply a scalar transformation to `Y`. Possible values:
     *   `opy = Op::In`   : Apply the idenity to the elements of `Y`.
     *   `opy = Op::Conj` : Apply `std::Conj` to the elements of `Y`.
     *
     * @param a A scalar.
     *
     * @param X A matrix of size `m` x `n`.
     *
     * @param ldX Leading dimension of `X`.
     *
     * @param b A scalar.
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
    
    template<
        Size_T M, Size_T N, Parallel_T parQ,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices_auto(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY,
        const Int m, const Int n, const Int thread_count
    )
    {
        using F_T = Scalar::Flag;
        
        auto job = [=]<F_T a_flag, F_T b_flat>()
        {
            combine_matrices<a_flag,b_flat,M,N,parQ,opx,opy>(
                a, X, ldX,
                b, Y, ldY,
                m, n, thread_count
             );
        };
        
        if( a == a_T(1) )
        {
            constexpr F_T a_flag = F_T::Plus;
            
            if( b == b_T(1) )
            {
                constexpr F_T b_flag = F_T::Plus;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(0) )
            {
                constexpr F_T b_flag = F_T::Zero;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(-1) )
            {
                constexpr F_T b_flag = F_T::Minus;
                job.template operator()<a_flag,b_flag>();
            }
            else
            {
                constexpr F_T b_flag = F_T::Generic;
                job.template operator()<a_flag,b_flag>();
            }
        }
        else if ( a == a_T(0) )
        {
            constexpr F_T a_flag = F_T::Zero;
            
            if( b == b_T(1) )
            {

                // Do nothing.
//                constexpr F_T b_flag = F_T::Plus;
//                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(0) )
            {
                constexpr F_T b_flag = F_T::Zero;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(-1) )
            {
                constexpr F_T b_flag = F_T::Minus;
                job.template operator()<a_flag,b_flag>();
            }
            else
            {
                constexpr F_T b_flag = F_T::Generic;
                job.template operator()<a_flag,b_flag>();
            }
        }
        else if ( a == a_T(-1) )
        {
            constexpr F_T a_flag = F_T::Minus;
            
            if( b == b_T(1) )
            {
                constexpr F_T b_flag = F_T::Plus;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(0) )
            {
                constexpr F_T b_flag = F_T::Zero;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(-1) )
            {
                constexpr F_T b_flag = F_T::Minus;
                job.template operator()<a_flag,b_flag>();
            }
            else
            {
                constexpr F_T b_flag = F_T::Generic;
                job.template operator()<a_flag,b_flag>();
            }
        }
        else
        {
            constexpr F_T a_flag = F_T::Generic;
            
            if( b == b_T(1) )
            {
                constexpr F_T b_flag = F_T::Plus;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(0) )
            {
                constexpr F_T b_flag = F_T::Zero;
                job.template operator()<a_flag,b_flag>();
            }
            else if ( b == b_T(-1) )
            {
                constexpr F_T b_flag = F_T::Minus;
                job.template operator()<a_flag,b_flag>();
            }
            else
            {
                constexpr F_T b_flag = F_T::Generic;
                job.template operator()<a_flag,b_flag>();
            }
        }
    }
    
    /*!
     * @brief Overload for fixed sizes. Automatic detection of `a_flag` and `b_flag` at runtime.
     */
    
    template<
        Size_T M, Size_T N,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY
    )
    {
        static_assert( M > VarSize, "" );
        static_assert( N > VarSize, "" );
        
        combine_matrices_auto<M,N,Sequential,opx,opy>(
            a, X, ldX,
            b, Y, ldY,
            M, N, Int(1)
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, parallel evaluation. Automatic detection of `a_flag` and `b_flag` at runtime.
     */
    template<
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY,
        const Int m, const Int n, const Int thread_count
    )
    {
        combine_matrices_auto<VarSize,VarSize,Parallel,opx,opy>(
            a, X, ldX,
            b, Y, ldY,
            m, n, thread_count
         );
    }
    
    
    /*!
     * @brief Overload for variable sizes, sequential evaluation. Automatic detection of `a_flag` and `b_flag` at runtime.
     */
    
    template<
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename X_T, typename b_T, typename Y_T, IntQ Int
    >
    TOOLS_FORCE_INLINE void combine_matrices(
        cref<a_T> a, cptr<X_T> X, const Int ldX,
        cref<b_T> b, mptr<Y_T> Y, const Int ldY,
        const Int m, const Int n
    )
    {
        combine_matrices_auto<VarSize,VarSize,Sequential,opx,opy>(
            a, X, ldX,
            b, Y, ldY,
            m, n, Int(1)
         );
    }
    
} // namespace Tools
