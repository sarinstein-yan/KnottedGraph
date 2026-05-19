#pragma once

namespace Tools
{
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE x_T total_buffer(
        cptr<x_T> x, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        {
            TOOLS_MAKE_FP_FAST()
            
            return DoReduce<N,parQ>(
                [x]( const Int i ) { return x[i]; },
                []( const x_T & value, x_T & result ) { result += value; },
                x_T(0),
                n, thread_count
            );
        }
    }
    
} // namespace Tools
