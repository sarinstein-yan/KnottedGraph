#pragma once

namespace Tools
{
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename a_T, typename x_T, IntQ Int, typename b_T, typename y_T
    >
    constexpr TOOLS_FORCE_INLINE void combine_scatter_write(
        cref<a_T> a, cptr<x_T> x,
        cref<b_T> b, mptr<y_T> y, cptr<Int> idx, Int n = static_cast<Int>(N), Int thread_count = 1
    )
    {
        check_sequential<parQ>( "combine_scatter_write", thread_count );
        
        using namespace Scalar;
        
        // This routine computes y[i] = a * x[i] + b * y[i].
        // Depending on the values of a_flag and b_flag, it takes several short cuts:
        // If a_flag == Flag::Zero,  then it assumes a = 0.
        // If a_flag == Flag::Plus,  then it assumes a = 1.
        // If a_flag == Flag::Minus, then it assumes a = -1.
        // If a_flag == Flag::Generic, then it assumes generic values for a.
        
        // If b_flag == Flag::Zero,  then it assumes b = 0.
        // If b_flag == Flag::Plus,  then it assumes b = 1.
        // If b_flag == Flag::Minus, then it assumes b = -1.
        // If b_flag == Flag::Generic, then it assumes generic values for b.

        static_assert( ComplexQ<y_T> || (RealQ<a_T> && RealQ<x_T> && RealQ<b_T>),
            "Fourth argument is real, but some of the other arguments are complex."
        );
        
        // We refrain from automagically casting `a` and `b` to the right precision because this is better done once before any loop that calls `combine_buffers`. Hence, we prefer a compile error here.
        
        static_assert(
            Prec<a_T> == Prec<y_T>,
            "Precisions of first and fourth argument do not coincide."
        );
        
        static_assert(
            Prec<b_T> == Prec<y_T>,
            "Precisions of third and fourth argument do not coincide."
        );
    
        {
            TOOLS_MAKE_FP_FAST()
            
            Do<N,parQ,Static>(
                [=]( const Int k )
                {
                    combine_scalars<a_flag,b_flag>( a, x[k], b, y[idx[k]] );
                },
                n, thread_count
            );
        }
    }

} // namespace Tools


