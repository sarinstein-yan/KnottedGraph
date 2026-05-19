#pragma once

namespace Tools
{
    /*!
     * @brief Computes `y = a * opx(x) + b * opy(y)`, where `x` and `y`
     * are vectors of size `n` and `a` and `b` are scalars.
     *
     * @tparam a_flag Compile-time knowledge for the scalar `a`.
     *
     * @tparam b_flag Compile-time knowledge for the scalar `b`.
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
     * @param x A vector of size `n`.
     *
     * @param b A scalar.
     *
     * @param y A vector of size `n`.
     *
     * @param n Number of entries.
     *
     * @param thread_count Number of threads to parallelize over.
     */
    
    // TODO: We might want to have also inc_x and inc_y variants.
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename x_T, typename b_T, typename y_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void combine_buffers(
        [[maybe_unused]] cref<a_T> a, cptr<x_T> x,
        [[maybe_unused]] cref<b_T> b, mptr<y_T> y,
        const Int n = static_cast<Int>(N),
        const Int thread_count = 1
    )
    {
        // This routine computes y[i] = a * x[i] + b * y[i].
        //
        // Depending on the values of a_flag and b_flag, it takes several short cuts:
        // If a_flag == Flag::Zero,  then it assumes a = 0.
        // If a_flag == Flag::Plus,  then it assumes a = 1.
        // If a_flag == Flag::Minus, then it assumes a = -1.
        // If a_flag == Flag::Generic, then it assumes generic values for `a`.
        
        // If b_flag == Flag::Zero,  then it assumes b = 0.
        // If b_flag == Flag::Plus,  then it assumes b = 1.
        // If b_flag == Flag::Minus, then it assumes b = -1.
        // If b_flag == Flag::Generic, then it assumes generic values for `b`.
        
        using namespace Scalar;
        
        check_sequential<parQ>( "combine_buffers", thread_count );
        
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
//        static_assert(
//            ComplexQ<y_T>
//            ||
//            Scalar::RealQ<decltype( Scalar::Op<a_flag,opx>( a_T(0), x_T(0) ) )>,
//            "Fourth argument is real, but some of the other arguments are complex."
//        );
        
        // We refrain from automagically casting `a` and `b` to the right precision because this is better done once before any loop that calls `combine_buffers`. Hence, we prefer a compile error here.
        
        static_assert(
            Prec<a_T> == Prec<x_T>,
            "Precisions of first and second argument do not coincide."
        );
        
        static_assert(
            Prec<b_T> == Prec<y_T>,
            "Precisions of third and fourth argument do not coincide."
        );
        

        // PROBLEM: If x == nullptr, which is allowed when a_flag = Zero,
        // then we should prevent dereferencing x.
        // Otherwise, with optimizations turned off, it could happen that nullptr
        // is dereferenced.
        if constexpr ( a_flag == Scalar::Flag::Zero )
        {
            // PROBLEM: If x == nullptr, which is allowed when a_flag = Zero, then we should prevent dereferencing x.
            // Otherwise, with optimizations turned off, it could happen that nullptr
            // is dereferenced.
            if constexpr ( (b_flag == Scalar::Flag::Plus) && (opy == Op::Id) )
            {
                // Do nothing;
            }
            else
            {
                modify_buffer<b_flag,N,parQ,opy>( b, y, n, thread_count );
            }
        }
        if constexpr ( (a_flag == Scalar::Flag::Plus) && (b_flag == Scalar::Flag::Zero) && (opx == Op::Id) )
        {
            copy_buffer<N,parQ>( x, y, n, thread_count );
        }
        else
        {
            TOOLS_MAKE_FP_FAST()

            Do<N,parQ,Static>(
                [=]( const Int i )
                {
                    combine_scalars<a_flag,b_flag,opx,opy>( a, x[i], b, y[i] );
                },
                n, thread_count
            );
        }
    }
    
    
    
    /*!
     * @brief Computes `z = a * opx(x) + b * opy(y)`, where `x`, `y` and `z`
     * are vectors of size `n` and `a` and `b` are scalars.
     *
     * @tparam a_flag Compile-time knowledge for the scalar `a`.
     *
     * @tparam b_flag Compile-time knowledge for the scalar `b`.
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
     * @param x A vector of size `n`.
     *
     * @param b A scalar.
     *
     * @param y A vector of size `n`.
     *
     * @param n Number of entries.
     *
     * @param thread_count Number of threads to parallelize over.
     */
    
    // TODO: We might want to have also inc_x, inc_y, inc_z variants.
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        Op opx = Op::Id, Op opy = Op::Id,
        typename a_T, typename x_T, typename b_T, typename y_T, typename z_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void combine_buffers3(
        [[maybe_unused]] cref<a_T> a, cptr<x_T> x,
        [[maybe_unused]] cref<b_T> b, cptr<y_T> y,
                                      mptr<z_T> z,
        const Int n = static_cast<Size_T>(N),
        const Int thread_count = Int(1)
    )
    {
        // This routine computes z[i] = a * x[i] + b * y[i].
        // Most importantly, this casts the result to the desired type.
        //
        // Depending on the values of a_flag and b_flag, it takes several short cuts:
        // If a_flag == Flag::Zero,  then it assumes a = 0.
        // If a_flag == Flag::Plus,  then it assumes a = 1.
        // If a_flag == Flag::Minus, then it assumes a = -1.
        // If a_flag == Flag::Generic, then it assumes generic values for a.
        
        // If b_flag == Flag::Zero,  then it assumes b = 0.
        // If b_flag == Flag::Plus,  then it assumes b = 1.
        // If b_flag == Flag::Minus, then it assumes b = -1.
        // If b_flag == Flag::Generic, then it assumes generic values for b.
        
        using namespace Scalar;
        
        check_sequential<parQ>( "combine_buffers3", thread_count );
        
        static_assert( (opx == Op::Id) || (opx == Op::Conj),
            "combine_buffers3: Only the values Op::Id and Op::Conj are allowed for opx."
        );
        
        static_assert( (opy == Op::Id) || (opy == Op::Conj),
            "combine_buffers3: Only the values Op::Id and Op::Conj are allowed for opy."
        );
        
        static_assert( ComplexQ<z_T> || (RealQ<a_T> && RealQ<x_T> && RealQ<b_T> && RealQ<y_T> ),
            "Last argument is real, but some of the other arguments are complex."
        );
        
        // We refrain from automagically casting `a` and `b` to the right precision because this is better done once before any loop that calls `combine_buffers`. Hence, we prefer a compile error here.
        
        static_assert(
            Prec<a_T> == Prec<x_T>,
            "Precisions of first and second argument do not coincide."
        );
        
        static_assert(
            Prec<b_T> == Prec<y_T>,
            "Precisions of third and fourth argument do not coincide."
        );
  
        // PROBLEM: If x == nullptr, which is allowed when a_flag = Zero,
        // then we should prevent dereferencing x.
        // Otherwise, with optimizations turned off, it could happen that nullptr
        // is dereferenced.
        if constexpr ( a_flag == Scalar::Flag::Zero )
        {
            // PROBLEM: If y == nullptr, which is allowed when (b_flag = Zero),
            // then we should prevent dereferencing y.
            // Otherwise, with optimizations turned off, it could happen that nullptr
            // is dereferenced.
            if constexpr ( (b_flag == Scalar::Flag::Zero) )
            {
                // z = 0;
                modify_buffer<Scalar::Flag::Zero,N,parQ,Op::Id>(
                    Scalar::Zero<z_T>, z, n, thread_count
                );
            }
            else
            {
                // z = b * opy(y);
                combine_buffers<b_flag,Scalar::Flag::Zero,N,parQ,opy,Op::Id>(
                    b, y, Scalar::Zero<z_T>, z, n, thread_count
                );
            }
        }
        else
        {
            if constexpr ( (b_flag == Scalar::Flag::Zero) )
            {
                // z = a * opx(x);
                
                combine_buffers<a_flag,Scalar::Flag::Zero,N,parQ,opx,Op::Id>(
                    a, x, Scalar::Zero<z_T>, z, n, thread_count
                );
            }
            else
            {
                TOOLS_MAKE_FP_FAST()
                
                // z = a * opx(x) + b * opx(y);
                Do<N,parQ,Static>(
                    [=]( const Int i )
                    {
                        combine_scalars<a_flag,b_flag,opx,opy>( a, x[i], b, y[i], z[i] );
                    },
                    n, thread_count
                );
            }
        }
    }

} // namespace Tools
