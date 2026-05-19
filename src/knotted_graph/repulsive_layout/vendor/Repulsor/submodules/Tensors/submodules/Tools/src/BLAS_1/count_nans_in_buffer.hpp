#pragma once

namespace Tools
{
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE Int count_nans_in_buffer(
        mptr<T> a,
        const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        check_sequential<parQ>( "count_nans_in_buffer", thread_count );
        
        return DoReduce<VarSize,parQ>(
            [=]( const Int i ) -> Int { return NaNQ(a[i]); },
            AddReducer<Int,Int>(), Scalar::Zero<Int>,
            n, thread_count
        );
    }
    
} // namespace Tools
