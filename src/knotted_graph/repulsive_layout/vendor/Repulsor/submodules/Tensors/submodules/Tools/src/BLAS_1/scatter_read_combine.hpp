#pragma once

namespace Tools
{
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename R_0, typename S_0, IntQ Int, typename R_1, typename S_1
    >
    constexpr TOOLS_FORCE_INLINE void scatter_read_combine(
        cref<R_0> a, cptr<S_0> x, cptr<Int> idx,
        cref<R_1> b, mptr<S_1> y,                 Size_T n = N, Size_T thread_count = 1
    )
    {
        check_sequential<parQ>( "scatter_read_combine", thread_count );
        
        using namespace Scalar;
        
        // This routine computes y[k] = a * x[idx[k]] + b * y[k].
        // Depending on the values of a_flag and b_flag, it takes several short cuts:
        // If a_flag == Flag::Zero,  then it assumes a = 0.
        // If a_flag == Flag::Plus,  then it assumes a = 1.
        // If a_flag == Flag::Minus, then it assumes a = -1.
        // If a_flag == Flag::Generic, then it assumes generic values for a.
        
        // If b_flag == Flag::Zero,  then it assumes b = 0.
        // If b_flag == Flag::Plus,  then it assumes b = 1.
        // If b_flag == Flag::Minus, then it assumes b = -1.
        // If b_flag == Flag::Generic, then it assumes generic values for b.

        static_assert( ComplexQ<S_1> || (RealQ<R_0> && RealQ<S_0> && RealQ<R_1>),
            "Fourth argument is real, but some of the other arguments are complex."
        );
        
        // We refrain from automagically casting `a` and `b` to the right precision because this is better done once before any loop that calls `combine_buffers`. Hence, we prefer a compile error here.
        
        static_assert(
            Prec<R_0> == Prec<S_1>,
            "Precisions of first and fourth argument do not coincide."
        );
        
        static_assert(
            Prec<R_1> == Prec<S_1>,
            "Precisions of third and fourth argument do not coincide."
        );
        
        if constexpr ( (b_flag == Flag::Zero) && (a_flag == Flag::Zero) )
        {
            zerofy_buffer<N,parQ>( y, n, thread_count );
        }
        else
        {
            TOOLS_MAKE_FP_FAST()
            
            Do<N,parQ>(
                [a,x,b,y,idx]( const Size_T k )
                {
                    combine_scalars<a_flag,b_flag>( a, x[idx[k]], b, y[k] );
                },
                n, thread_count
            );
        }
    }

} // namespace Tools

