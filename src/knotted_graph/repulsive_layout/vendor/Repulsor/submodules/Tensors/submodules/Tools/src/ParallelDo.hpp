#pragma once

#include <thread>
#include <future>

namespace Tools
{
    template<typename F, typename Int>
    concept Doable = IntQ<Int> && (std::invocable<F,Int> || std::invocable<F,Int,Int>);
    
    
    // Executes the function `fun` of the form `[]( const Int thread ) -> S {...}` parallelized over `thread_count` threads.
    template<typename F, IntQ Int> requires std::invocable<F,Int>
    TOOLS_FORCE_INLINE void ParallelDo( F && fun, const Int thread_count )
    {
        if( thread_count <= Int(1) )
        {
            std::invoke( fun, Int(0) );
        }
        else
        {
            // We want to avoid the profiler messing up, so I silence it for the duration of the parallel region.
            Profiler::Blocker blocker;
            
            std::vector<std::future<void>> futures ( ToSize_T(thread_count) );
            
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                futures[static_cast<Size_T>(thread)] = std::async( std::launch::async, std::forward<F>(fun), thread );
            }
            
            for( auto & future : futures ) { future.get(); }
        }
    }
    
    
    

    // Executes the function `f` of the form `[]( Int thread, Int i ) {...}` or `[]( Int i ) {...}` over the range [begin,end[, using just one thread, but using the thread id given by the argument `thread`.
//    template<typename F, IntQ Int = Size_T>
//    requires std::invocable<F,Int> || std::invocable<F,Int,Int>
//    TOOLS_FORCE_INLINE void SequentialDo( F && f, const Int begin, const Int end, const Int thread )
//    {
//        static_assert(function_traits<F>::arity >= 1, "");
//        static_assert(function_traits<F>::arity <= 2, "");
//
//        for( Int i = begin; i < end; ++i )
//        {
//            if constexpr( function_traits<F>::arity == 2 )
//            {
//                std::invoke(f, thread, i);
//            }
//            else
//            {
//                std::invoke(f, i);
//            }
//        }
//    }
    
    template<typename F, IntQ Int = Size_T> requires std::invocable<F,Int>
    TOOLS_FORCE_INLINE void SequentialDo( F && f, const Int begin, const Int end, const Int thread )
    {
        (void)thread;
        for( Int i = begin; i < end; ++i ) { std::invoke(f, i); }
    }
    
    template<typename F, IntQ Int = Size_T> requires std::invocable<F,Int,Int>
    TOOLS_FORCE_INLINE void SequentialDo( F && f, const Int begin, const Int end, const Int thread )
    {
        for( Int i = begin; i < end; ++i ) { std::invoke(f, thread, i); }
    }
    
    
//    namespace Details
//    {
//        template<>
//    }
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) {...}` or `[]( Int i ) {...}` over the range [begin,end[, using `thread_count` threads.
    template<typename F, IntQ Int> requires Doable<F,Int>
    TOOLS_FORCE_INLINE void ParallelDo_Static(
        F && f, const Int begin, const Int end, const Int thread_count
    )
    {
        ParallelDo(
            [&f,begin,end,thread_count]( const Int thread )
            {
                const Int i_begin = begin + JobPointer( end - begin, thread_count, thread     );
                const Int i_end   = begin + JobPointer( end - begin, thread_count, thread + 1 );
                
                SequentialDo( std::forward<F>(f), i_begin, i_end, thread );
            },
            thread_count
        );
    }
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) {...}` or `[]( Int i ) {...}` over the range [begin,end[, using `thread_count` threads.
    template<typename F, IntQ Int = Size_T> requires Doable<F,Int>
    TOOLS_FORCE_INLINE void ParallelDo_Dynamic(
        F && f, const Int begin, const Int end, const Int inc, const Int thread_count
    )
    {
        if( end <= begin ) { return; }
        
        // We want to avoid the profiler messing up, so we silence it for the duration of the parallel region.
        Profiler::Blocker blocker;
        
        // Global index that is supposed to run from begin to end; shared by all threads.
        Int iter = begin;
        // A mutex to synchronize iter.
        std::mutex iter_mutex;
        
        // A vector to hold the workers.
        std::vector<std::future<void>> futures ( ToSize_T(thread_count) );
        
        // Lambda-function representing a single worker.
        auto worker = [&f,&iter,&iter_mutex,end,inc]( const Int thread )
        {
            bool done = false;
            
            // This worker is supposed to repeatedly check whether `iter` has run out of bounds.
            while( !done )
            {
                Int i_begin = end;
                Int i_end   = end;
                
                {
                    const std::lock_guard<std::mutex> lock_iter(iter_mutex);
                
                    if( iter < end )
                    {
                        // Iterator `iter` is not out of bounds. Assign a new chunk to worker.
                        i_begin = iter;
                        iter    = Min( iter + inc, end );
                        i_end   = iter;
                    }
                    else
                    {
                        // Nothing left to do. Stop here.
                        done = true;
                        
                        TOOLS_DEBUG_PRINT( "ParallelDo_Dynamic: " + ToString(thread) + " done." );
                        
                        return;
                    }
                }
                
                TOOLS_DEBUG_PRINT( "ParallelDo_Dynamic: thread " + ToString(thread) + " processes tasks [" + ToString(i_begin) + "," + ToString(i_end)  + "[." );
                // Let the worker process its assigned chunk.
                SequentialDo( std::forward<F>(f), i_begin, i_end, thread );

//                for( Int i = i_begin; i < i_end; ++i )
//                {
//                    if constexpr( function_traits<F>::arity == 2 )
//                    {
//                        std::invoke(f, thread, i);
//                    }
//                    else
//                    {
//                        std::invoke(f, i);
//                    }
//                }
            }
        };
        
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            futures[static_cast<Size_T>(thread)] = std::async(std::launch::async, worker, thread);
        }
        
        for( auto & future : futures ) { future.get(); }
    }
    
} // namespace Tools
