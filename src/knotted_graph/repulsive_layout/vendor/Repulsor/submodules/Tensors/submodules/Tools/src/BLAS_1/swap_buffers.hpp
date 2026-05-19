#pragma once

namespace Tools
{
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void swap_buffers(
        mptr<T> x, mptr<T> y,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        check_sequential<parQ>( "swap_buffers", thread_count );
        
        if constexpr ( N <= VarSize )
        {
            if constexpr ( parQ == Sequential )
            {
                std::swap_ranges( x, &x[n], y );
            }
            else
            {
                if( thread_count <= Scalar::One<Int> )
                {
                    std::swap_ranges( x, &x[n], y );
                }
                else
                {
                    ParallelDo(
                        [=]( const Int thread )
                        {
                            const Int begin = JobPointer(n,thread_count,thread  );
                            const Int end   = JobPointer(n,thread_count,thread+1);

                            std::swap_ranges( &x[begin], &x[end], y );
                        },
                        thread_count
                    );
                }
            }
        }
        else
        {
            for( Int i = 0; i < N; ++i )
            {
                using std::swap;
                
                swap( x[i], y[i] );
            }
        }
    }
    
} // namespace Tools
