#pragma once

namespace Tools
{
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void move_buffer(
        cptr<T> source, mptr<T> target, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        check_sequential<parQ>( "move_buffer", thread_count );

        if constexpr ( N <= VarSize )
        {
            if constexpr ( parQ == Sequential )
            {
                std::move( &source[0], &source[n], &target[0] );
            }
            else
            {
                if( thread_count <= Int(1) )
                {
                    std::move( &source[0], &source[n], &target[0] );
                }
                else
                {
                    ParallelDo(
                        [=]( const Int thread )
                        {
                            const Int begin = JobPointer(n,thread_count,thread  );
                            const Int end   = JobPointer(n,thread_count,thread+1);

                            std::move( &source[begin], &source[end], &target[begin] );
                        },
                        thread_count
                    );
                }
            }
        }
        else
        {
            std::move( &source[0], &source[n], &target[0] );
        }
    }
    
} // namespace Tools
