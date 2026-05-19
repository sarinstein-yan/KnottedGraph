#pragma once

namespace Tools
{
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename y_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void add_to_buffer(
        cptr<x_T> x, mptr<y_T> y,
        const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        combine_buffers<Scalar::Flag::Plus,Scalar::Flag::Plus,N,parQ,Op::Id,Op::Id>(
            Scalar::One<y_T>, x,
            Scalar::One<y_T>, y,
            n,thread_count
        );
    }
    
} // namespace Tools
