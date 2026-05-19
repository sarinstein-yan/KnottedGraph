#pragma once

namespace Tools
{
    enum struct Dynamic_T : bool
    {
        True  = true,
        False = false
    };
    
    static constexpr Dynamic_T Dynamic = Dynamic_T::True;
    static constexpr Dynamic_T Static  = Dynamic_T::False;
    
    std::string ToString( const Dynamic_T & parQ )
    {
        switch (parQ)
        {
            case Dynamic: return "Dynamic";
            case Static:  return "Static";
        }
    }
    
    template<IntQ Int>
    struct DoRange
    {
        Int begin = 0;
        Int end   = 0;
        
        DoRange( Int n )
        :   end { n }
        {}
        
        template<IntQ I>
        DoRange( I a, Int b )
        :   begin { static_cast<Int>(a) }
        ,   end   { b }
        {}
    };
    
    
    template<Dynamic_T dynQ>
    void constexpr not_dynamic()
    {
        static_assert(dynQ == Static, "This method ignores dynamic scheduling; it is always static.");
    }
    
    template<Parallel_T parQ, IntQ Int >
    void check_sequential( std::string tag, Int thread_count )
    {
#ifdef TOOLS_DEBUG
        if constexpr ( parQ == Sequential )
        {
            if ( thread_count > Int(1) )
            {
                wprint(std::string("check_sequential: method ") + tag + " is marked as sequential, but more than one thread is requested.");
            }
        }
#else
        (void)tag;
        (void)thread_count;
#endif
    }

    //###########################################################
    //##  Do
    //###########################################################
    
    // Executes the function `fun` of the form `[]( Int i ) {...}` or `[]( thread i, Int i )  {...}` on the range [begin,end[, parallelized over `thread_count` threads.
    // Parallelization will be turned off if `parQ` is set to `Sequential`. This guarantees that no threaded code is generated.
    template<Parallel_T parQ = Sequential, Dynamic_T dynQ = Static, typename F, IntQ Int>
    requires Doable<F,Int>
    TOOLS_FORCE_INLINE void Do( F && f, cref<DoRange<Int>> range, const Int thread_count = 1 )
    {
        if constexpr (parQ == Parallel)
        {
            if( thread_count <= Int(1) )
            {
                SequentialDo( std::forward<F>(f), range.begin, range.end, Int(0) );
            }
            else
            {
                if constexpr ( dynQ == Dynamic )
                {
                    ParallelDo_Dynamic(std::forward<F>(f), range.begin, range.end, Int(1), thread_count);
                }
                else
                {
                    ParallelDo_Static(std::forward<F>(f), range.begin, range.end, thread_count);
                }
            }
        }
        else // if constexpr (parQ == Sequential)
        {
            if( thread_count > Int(1) )
            {
                wprint("Do was called in sequential mode, but thread_count = " + ToString(thread_count)+  " > 1. Beware of unintended behavior.");
            }
            SequentialDo( std::forward<F>(f), range.begin, range.end, Int(0) );
        }
    }
    
    // Executes the function `fun` of the form `[]( Int i ) {...}` or `[]( thread i, Int i )  {...}` on the range [0,n[, parallelized over `thread_count` threads.
    // This allows also loops with constant trip counts `N`; if `N > 0`, then the loop will be unrolled and processed sequentially.
    // Parallelization will be turned off if `parQ` is set to `Sequential`. This guarantees that no threaded code is generated.
    template<
        Size_T N, Parallel_T parQ = Sequential, Dynamic_T dynQ = Static, typename F, IntQ Int = Size_T
    >
    requires Doable<F,Int>
    TOOLS_FORCE_INLINE void Do( F && f, const Int n = static_cast<Int>(N), const Int thread_count = 1)
    {
        if constexpr ( N <= VarSize )
        {
            Do<parQ,dynQ>( std::forward<F>(f), {Int(0), n}, thread_count );
        }
        else
        {
            if( thread_count > Int(1) )
            {
                wprint("Do was called with fixed size N = " + ToString(N)+  " > 1, but thread_count = " + ToString(thread_count)+  " > 1. Beware of unintended behavior.");
            }
            
            constexpr Int N_ = static_cast<Int>(N);
            for( Int i = 0; i < N_; ++i )
            {
                if constexpr ( std::invocable<F,Int,Int> )
                {
                    std::invoke(f,Int(0),i);
                }
                else if constexpr ( std::invocable<F,Int> )
                {
                    std::invoke(f,i);
                }
            }
        }
    }
    
    template<Parallel_T parQ = Sequential, Dynamic_T dynQ = Static, typename F, IntQ Int = Size_T>
    requires Doable<F,Int>
    TOOLS_FORCE_INLINE void Do( F && f, const Int n, const Int thread_count = 1 )
    {
        Do<parQ,dynQ>( std::forward<F>(f), {Int(0), n}, thread_count );
    }
    
    
    // Executes the function `f` of the form `[]( Int thread, Int i ) {...}` or `[]( Int i ) {...}` over the range [begin,end[, using `thread_count` threads.
    template<Parallel_T parQ = Parallel, typename F, IntQ Int>
    requires Doable<F,Int>
    TOOLS_FORCE_INLINE void Do( F && f, cref<JobPointers<Int>> job_ptr )
    {
        if constexpr ( parQ == Parallel )
        {
            ParallelDo(
                [&f,&job_ptr]( const Int thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];

                    SequentialDo( std::forward<F>(f), i_begin, i_end, thread );
                },
                job_ptr.ThreadCount()
            );
        }
        else
        {
            if( job_ptr.ThreadCount() > Int(1) )
            {
                wprint("DoReduce was called in sequential mode, but job_ptr.ThreadCount() = " + ToString(job_ptr.ThreadCount())+  " > 1. Beware of unintended behavior.");
            }
            
            Do<Sequential>( std::forward<F>(f), job_ptr.JobCount() );
        }
    }

} // namespace Tools
