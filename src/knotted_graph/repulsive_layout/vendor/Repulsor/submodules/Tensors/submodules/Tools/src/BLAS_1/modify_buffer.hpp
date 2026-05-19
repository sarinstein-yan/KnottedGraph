#pragma once

namespace Tools
{
    
    /*!
     * @brief Computes `x = a * op(x)`, where `x` is a vector of size `n`
     * and `a` is a scalar.
     *
     * @tparam a_flag Compile-time knowledge for the scalar `a`.
     *
     * @tparam N Compile-time knowledge of the number of columns.
     *   `N > 0` : Use fixed-size loops that can be optimized by the compiler.
     *   `N = 0` : Use variable the run-time column count specified by argument `n`.
     *
     * @tparam parQ Whether to parallelize or not.
     *
     * @tparam op Apply a scalar transformation to `X`. Possible values:
     *   `op = Op::In`   : Apply the identity to the elements of `X`.
     *   `op = Op::Conj` : Apply `std::Conj` to the elements of `X`.
     *
     * @param a A scalar.
     *
     * @param x A vector of size `n`.
     *
     * @param n Number of entries.
     *
     * @param thread_count Number of threads to parallelize over.
     */
    
    template<
        Scalar::Flag a_flag, Size_T N = VarSize, Parallel_T parQ = Sequential,
        Op op = Op::Id,
        typename a_T, typename x_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void modify_buffer(
        [[maybe_unused]] cref<a_T> a, mptr<x_T> x,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        // This routine computes y[i] = a * x[i].
        //
        // Depending on the value of a_flag, it takes several short cuts:
        // If a_flag == Flag::Zero,  then it assumes a = 0.
        // If a_flag == Flag::Plus,  then it assumes a = 1.
        // If a_flag == Flag::Minus, then it assumes a = -1.
        // If a_flag == Flag::Generic, then it assumes generic values for `a`.
        
        using namespace Scalar;
        
        check_sequential<parQ>( "modify_buffer", thread_count );
        
        static_assert( (op == Op::Id) || (op == Op::Conj) || (op == Op::Re) || (op == Op::Im),
            "combine_buffers: Only the values Op::Id, Op::Conj, Op::Re, and Op::Im are allowed for op."
        );
        
        static_assert( ComplexQ<x_T> || RealQ<a_T>,
            "Second argument is real, but first of the other arguments are not."
        );
        
        // We refrain from automagically casting `a` to the right precision because this is better done once before any loop that calls `modify_buffer`. Hence, we prefer a compile error here.
        
        static_assert(
            Prec<a_T> == Prec<x_T>,
            "Precisions of first and second argument do not coincide."
        );
        

        // PROBLEM: If x == nullptr, which is allowed when
        //
        //      a_flag = One and op == Op::Id,
        //
        // then we should better prevent dereferencing x.
        // Otherwise, with optimizations turned off, it could happen that nullptr
        // is dereferenced.
        if constexpr ( (a_flag == Scalar::Flag::Plus) && (op == Op::Id) )
        {
            // Do nothing;
        }
        else
        {
            TOOLS_MAKE_FP_FAST()
            
            Do<N,parQ,Static>(
                [x,a]( const Int i )
                {
                    x[i] = Scalar::Op<a_flag,op>(a,x[i]);
                },
                n, thread_count
            );
        }
    }

} // namespace Tools

