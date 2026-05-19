#pragma once

namespace Tools
{
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void fill_buffer(
        mptr<T> a, const T init, const Int n = static_cast<Size_T>(N), const Int thread_count = 1
    )
    {
        check_sequential<parQ>( "fill_buffer", thread_count );
        
        if constexpr ( N <= VarSize )
        {
            if constexpr ( parQ == Sequential )
            {
                std::fill( &a[0], &a[n], init );
            }
            else
            {
                if( thread_count <= Int(1) )
                {
                    std::fill( &a[0], &a[n], init );
                }
                else
                {
                    ParallelDo(
                        [=]( const Int thread )
                        {
                            const Int begin = JobPointer(n,thread_count,thread  );
                            const Int end   = JobPointer(n,thread_count,thread+1);
                            
                            std::fill( &a[begin], &a[end], init );
                        },
                        thread_count
                    );
                }
            }
        }
        else
        {
            static_assert(N > VarSize,"");
            static_assert(parQ == Sequential,"");
            
            std::fill( &a[0], &a[N], init );
        }
        
    }
    
} // namespace Tools
