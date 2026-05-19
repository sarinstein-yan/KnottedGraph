#pragma once

namespace Tools
{
    constexpr auto AndReducer()
    {
        return []( cref<bool> v, mref<bool> r ) { r = r && v; };
    }

    constexpr auto OrReducer()
    {
        return []( cref<bool> v, mref<bool> r ) { r = r || v; };
    }
    
    template<typename S, typename T>
    constexpr auto AddReducer()
    {
        return []( cref<S> v, mref<T> r ) { r += static_cast<T>(v); };
    }
    
    template<typename S, typename T>
    constexpr auto TimesReducer()
    {
        return []( cref<S> v, mref<T> r ) { r *= static_cast<T>(v); };
    }
    
    //###########################################################
    //##  DoReduce
    //###########################################################
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) -> S  {...}` or `[]( Int i ) -> S {...}` over the range [range.begin,range,end[, using `thread_count` threads.
    // Each thread runs the reducer function `r` of the form `[]( Int thread, Int i, cref<S>, mref<T> ) {...}` or `[]( cref<S>, mref<T> ) {...}` over the the results, initialized by `init` to create one result for the thread.
    // Then the reducer `r` is run over all results of all threads.
    //  CAUTION: `init` is assumed to be a NEUTRAL ELEMENT for the reduction. Each thread is initialized by it!
    template<Parallel_T parQ = Sequential, typename F, typename R, typename T, IntQ Int>
    requires DoReducible<F,R,T,Int>
    [[nodiscard]] TOOLS_FORCE_INLINE T DoReduce(
        F && f, R && r, cref<T> init, const DoRange<Int> & range, const Int thread_count = 1
    )
    {
        if constexpr (parQ == Parallel)
        {
            if( thread_count <= Int(1) )
            {
                return SequentialDoReduce(
                    std::forward<F>(f), std::forward<R>(r), init, range.begin, range.end, Int(0)
                );
            }
            else
            {
                return ParallelDoReduce_Static(
                    std::forward<F>(f), std::forward<R>(r), init, range.begin, range.end, thread_count
                );
            }
        }
        else // if constexpr (parQ == Sequential)
        {
            
            return SequentialDoReduce(
                std::forward<F>(f), std::forward<R>(r), init, range.begin, range.end, Int(0)
            );
        }
    }
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) -> S  {...}` or `[]( Int i ) -> S {...}` over the range [0,n[, using `thread_count` threads.
    // Each thread runs the reducer function `r` of the form `[]( Int thread, Int i, cref<S>, mref<T> ) {...}` or `[]( cref<S>, mref<T> ) {...}` over the the results, initialized by `init` to create one result for the thread.
    // Then the reducer `r` is run over all results of all threads.
    //  CAUTION: `init` is assumed to be a NEUTRAL ELEMENT for the reduction. Each thread is initialized by it!
    // This supports also loops with constant trip counts `N`; if `N > 0`, then the loop will be unrolled and processed sequentially.
    template<Parallel_T parQ = Sequential, typename T, typename F, typename R, IntQ Int>
    requires DoReducible<F,R,T,Int>
    [[nodiscard]] TOOLS_FORCE_INLINE T DoReduce(
        F && f, R && r, cref<T> init, const Int n, const Int thread_count = 1
    )
    {
        return DoReduce<parQ>(std::forward<F>(f), std::forward<R>(r), init, {Int(0),n}, thread_count);
    }
    
    template<Size_T N, Parallel_T parQ = Sequential,
        typename T, typename F, typename R, IntQ Int = Size_T
    >
    requires DoReducible<F,R,T,Int>
    [[nodiscard]] TOOLS_FORCE_INLINE T DoReduce(
        F && f, R && r, cref<T> init, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        static_assert(function_traits<F>::arity <= 2, "");
        constexpr bool f_wants_threadQ = (function_traits<F>::arity == 2);
        
        static_assert(function_traits<R>::arity <= 3, "");
        constexpr bool r_wants_threadQ = (function_traits<R>::arity == 3);
        
        if constexpr ( N <= VarSize )
        {
            return DoReduce<parQ>( std::forward<F>(f), std::forward<R>(r), init, {Int(0),n}, thread_count );
        }
        else
        {
            constexpr Int N_ = static_cast<Int>(N);
            T result {init};
            const Int thread = 0;
            for( Int i = 0; i < N_; ++i )   // We want to exploit trip know at compile-time.
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
    }
    
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) -> S  {...}` or `[]( Int i ) -> S {...}` over the range [0,n[, using `thread_count` threads.
    // Each thread runs the reducer function `r` of the form `[]( Int thread, Int i, cref<S>, mref<T> ) {...}` or `[]( cref<S>, mref<T> ) {...}` over the the results, initialized by `init` to create one result for the thread.
    // Then the reducer `r` is run over all results of all threads.
    //  CAUTION: `init` is assumed to be a NEUTRAL ELEMENT for the reduction. Each thread is initialized by it!
    template<Parallel_T parQ = Parallel, typename F, typename R, typename T, IntQ Int>
    requires DoReducible<F,R,T,Int>
    TOOLS_FORCE_INLINE void DoReduce( F && f, R && r, cref<T> init, cref<JobPointers<Int>> job_ptr )
    {
        if constexpr ( parQ == Parallel )
        {
            ParallelDoReduce(
                [&f,&r,&init,&job_ptr]( const Int thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];

                    return SequentialDoReduce( std::forward<F>(f), std::forward<R>(r), init, i_begin, i_end, thread );
                },
                std::forward<R>(r), init,
                job_ptr.ThreadCount()
            );
        }
        else
        {
            if( job_ptr.ThreadCount() > Int(1) )
            {
                wprint("DoReduce was called in sequential mode, but job_ptr.ThreadCount() = " + ToString(job_ptr.ThreadCount())+  " > 1. Beware of unintended behavior.");
            }
            
            DoReduce<Sequential>( std::forward<F>(f), job_ptr.JobCount() );
        }
    }


} // namespace Tools
