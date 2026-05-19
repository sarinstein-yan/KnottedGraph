#pragma once

namespace Tools
{
    template<typename F, typename R, typename T, typename Int>
    concept DoReducible = IntQ<Int> && (
        (
            std::invocable<F,Int> && (
                std::invocable<R, std::invoke_result_t<F,Int>, T&>
                ||
                std::invocable<R, Int, std::invoke_result_t<F,Int>, T&>
            )
        )
        ||
        (
            std::invocable<F,Int,Int> && (
                std::invocable<R, std::invoke_result_t<F,Int,Int>, T&>
                ||
                std::invocable<R, Int, std::invoke_result_t<F,Int,Int>, T&>
            )
        )
    );
    
    
    // Executes the function `f` of the `form []( Int thread ) {...}` parallelized over `thread_count` threads.
    // Afterwards, reduces with the reducer function `r` of the form `[]( Int thread, cref<S> value, mref<T> result ) {...}` or `[]( cref<S> value, mref<T> result ) {...}`.
    template<typename F, typename R, typename T, IntQ Int> requires DoReducible<F,R,T,Int>
    TOOLS_FORCE_INLINE T ParallelDoReduce( F && f, R && r, cref<T> init, const Int thread_count )
    {
        T result (init);
        
        if( thread_count <= Int(1) )
        {
            if constexpr ( std::invocable<F,Int,Int> )
            {
                std::invoke( r, Int(0), f(Int(0)), result );
            }
            else if constexpr ( std::invocable<F,Int> )
            {
                std::invoke( r, f(Int(0)), result );
            }
        }
        else
        {
            // I want to avoid the profiler messing up, so I silence it for the duration of the parallel region.
            Profiler::Blocker blocker;
            
            std::vector<std::future<T>> futures ( ToSize_T(thread_count) );
            
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                futures[static_cast<Size_T>(thread)] = std::async( std::launch::async, std::forward<F>(f), thread );
            }
            
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                if constexpr ( function_traits<R>::arity == 3 )
                {
                    std::invoke( r, thread, futures[static_cast<Size_T>(thread)].get(), result );
                }
                else
                {
                    std::invoke( r, futures[static_cast<Size_T>(thread)].get(), result );
                }
            }
        }
        
        return result;
    }
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) -> S  {...}` or `[]( Int i ) -> S {...}` over the range [begin,end[, using just one thread, but using the thread id given by the argument `thread`.
    // Then it runs the reducer function `r` of the form `[]( Int thread, Int i, cref<S>, mref<T> ) {...}` or `[]( cref<S>, mref<T> ) {...}` over the results and the value `init` to create one result that is returned.
    template<typename F, typename R, typename T, IntQ Int = Size_T> requires DoReducible<F,R,T,Int>
    TOOLS_FORCE_INLINE T SequentialDoReduce(
        F && f, R && r, cref<T> init, const Int begin, const Int end, const Int thread
    )
    {
        constexpr bool f_wants_threadQ = (function_traits<F>::arity == 2);
        constexpr bool r_wants_threadQ = (function_traits<F>::arity == 3);
        
        T result {init};

        for( Int i = begin; i < end; ++i )
        {
            if constexpr( f_wants_threadQ )
            {
                if constexpr( r_wants_threadQ )
                {
                    std::invoke( r, thread, std::invoke(f,thread,i), result );
                }
                else
                {
                    std::invoke( r, std::invoke(f,thread,i), result );
                }
            }
            else
            {
                if constexpr( r_wants_threadQ )
                {
                    std::invoke( r, thread, std::invoke(f,i), result );
                }
                else
                {
                    std::invoke( r, std::invoke(f,i), result );
                }
            }
        }
        return result;
    }
    
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) -> S  {...}` or `[]( Int i ) -> S {...}` over the range [begin,end[, using `thread_count` threads.
    // Each thread runs the reducer function `r` of the form `[]( Int thread, Int i, cref<S>, mref<T> ) {...}` or `[]( cref<S>, mref<T> ) {...}` over the the results, initialized by `init` to create one result for the thread.
    // Then the reducer `r` is run over all results of all threads.
    //  CAUTION: `init` is assumed to be a NEUTRAL ELEMENT for the reduction. Each thread is initialized by it!
    template<typename F, typename R, typename T, IntQ Int> requires DoReducible<F,R,T,Int>
    TOOLS_FORCE_INLINE T ParallelDoReduce_Static(
        F && f, R && r, cref<T> init,
        const Int begin, const Int end, const Int thread_count
    )
    {
        return ParallelDoReduce(
            [&f,&r,begin,end,thread_count,&init]( const Int thread ) -> T
            {
                const Int i_begin = begin + JobPointer( end - begin, thread_count, thread     );
                const Int i_end   = begin + JobPointer( end - begin, thread_count, thread + 1 );

                return SequentialDoReduce(
                    std::forward<F>(f), std::forward<R>(r), init, i_begin, i_end, thread
                );
            },
            std::forward<R>(r), init,
            thread_count
        );
    }
}
