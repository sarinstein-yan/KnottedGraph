#pragma once

namespace Tools
{
    
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void iota_buffer(
        mptr<T> a,
        const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        check_sequential<parQ>( "iota_buffer", thread_count );
        
        Do<N,parQ>(
            [a]( const Int i ) { a[i] = static_cast<T>(i); },
            n, thread_count
        );
    }
    
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename T
    >
    TOOLS_FORCE_INLINE constexpr void fill_range_buffer(
        mptr<T> a,
        const T start,
        const Size_T n = N, const Size_T thread_count = 1
    )
    {
        check_sequential<parQ>( "fill_range_buffer", thread_count );
        
        Do<N,parQ>(
            [a,start]( const Size_T i ) { a[i] = start + static_cast<T>(i); },
            n, thread_count
        );
    }
    
} // namespace Tools
