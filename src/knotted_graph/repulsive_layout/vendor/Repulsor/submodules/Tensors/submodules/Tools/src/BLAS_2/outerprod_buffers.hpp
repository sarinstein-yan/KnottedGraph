#pragma once

namespace Tools
{
    // TODO: We might want to have also inc_x, inc_y, inc_z variants.
    
    /*! Computes C = a * op_x(x) * op_y(y)^T + b * C, where op_x and op_y may either be `Op::Id` or `Op::Conj`.
     *
     *  @tparam M If `M` > 0, then `M` is interpreted as number of elements in `x`. Otherwise the argument `m` is used for that.
     *  @tparam N If `N` > 0, then `N` is interpreted as number of elements in `y`. Otherwise the argument `n` is used for that.
     *  @tparam a_flag Indicates whether `a` is to be expected to be generic or can be assumed to be `+1`, `-1`, or `0`.
     *  @tparam b_flag Indicates whether `b` is to be expected to be generic or can be assumed to be `+1`, `-1`, or `0`.
     *  @tparam op_x Indicates whether `x` is supposed to be conjugated or not.
     *  @tparam op_y Indicates whether `y` is supposed to be conjugated or not.
     *
     *  @param a See formula above.
     *  @param x A vector supposed to be at least of length `M` (or `m` if `M <= 0`).
     *  @param y A vector supposed to be at least of length `N` (or `n` if `N <= 0`).
     *  @param b See formula above.
     *  @param C A matrix supposed to be at least size length `M` x `N` (or `m` x `N`, `M` x `n` or `m` x `n`, depending on whether `M <= 0` or `N <= 0`.
     *  @param ldC Leading dimension of matrix z
     */
    
    template <
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T M = VarSize, Size_T N = VarSize,
        Parallel_T parQ = Sequential,
        Op op_x = Op::Id, Op op_y = Op::Id,
        typename a_T,  typename x_T, typename y_T,
        typename b_T, typename C_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void outerprod_buffers(
        cref<a_T> a, cptr<x_T> x, cptr<y_T> y,
        cref<b_T> b, mptr<C_T> C, const Int ldC,
        const Int m = static_cast<Int>(M),
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        static_assert( NotTransposedQ(op_x), "" );
        static_assert( NotTransposedQ(op_y), "" );
        
        constexpr Scalar::Flag x_flag =
            (a_flag == Scalar::Flag::Zero)
            ? Scalar::Flag::Zero
            : Scalar::Flag::Generic;
        
        if constexpr (
            (a_flag == Scalar::Flag::Zero) && (b_flag == Scalar::Flag::Plus)
        )
        {
            // Nothing to do in this case.
            return;
        }
        
        // TODO: For large matrices  ( (M <= 0) && (N <= 0)) we should do this in smaller blocks.
        
        // TODO: If for fixed sizes ((M > 0) && (N > 0)), we might also want to consider using mat_T.
        
//        print("C");
//        
//        TOOLS_DUMP(x_flag)
//        TOOLS_DUMP(b_flag)
//        TOOLS_DUMP(a)
//        TOOLS_DUMP(b)
//        TOOLS_DUMP(M)
//        TOOLS_DUMP(N)
//        TOOLS_DUMP(m)
//        TOOLS_DUMP(n)
//
        
        Do<M,parQ>(
            [=](const Int i )
            {
                combine_buffers<x_flag,b_flag,N,Sequential,op_y>(
                    Scalar::Op<a_flag,op_x>(a,x[i]), y, b, &C[ldC * i], n
                );
            },
            m, thread_count
        );
    }
        
} // namespace Tools
