#pragma once

namespace Tools
{
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename a_T, typename x_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void scale_buffer(
        cref<a_T> a, mptr<x_T> x,
        const Int n = static_cast<Int>(N), const Int thread_count = Int(1)
    )
    {
        modify_buffer<Scalar::Flag::Generic,N,parQ,Op::Id>(a,x,n,thread_count);
    }
    
} // namespace Tools
