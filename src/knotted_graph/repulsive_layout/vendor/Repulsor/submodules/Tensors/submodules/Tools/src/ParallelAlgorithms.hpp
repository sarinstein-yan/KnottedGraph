#pragma once

#include <numeric>

namespace Tools
{
    enum class Parallel_T : bool
    {
        True  = true,
        False = false
    };
    
    static constexpr Parallel_T Parallel   = Parallel_T::True;
    static constexpr Parallel_T Sequential = Parallel_T::False;
    
    std::string ToString( const Parallel_T & parQ )
    {
        switch (parQ)
        {
            case Parallel:   return "Parallel";
            case Sequential: return "Sequential";
        }
    }
        template<IntQ Int, Parallel_T parQ, IntQ I>
    constexpr inline Int TruncateThreadCount( const I thread_count )
    {
        return (parQ == Parallel) ? Ramp_1(static_cast<Int>(thread_count)) : Int(1);
    }
    
    
    template<typename T, IntQ I, typename F>
    inline void parallel_prefix_scan(
        cptr<T>  a,
        mptr<T>  b,
        const I n,
        const F & f,
        const I thread_count
    )
    {
        // Assumint that f is commutative and (nearly) associative, we compute
        //
        //    b[0] = a[0];
        //    b[1] = f(b[1], a[1]);
        //    b[2] = f(b[1], a[2]);
        //    b[3] = f(b[2], a[3]);
        //    b[3] = f(b[3], a[4]);
        //    ...
        
        std::vector<T> S_buffer ( thread_count+1 );
        
        mptr<T> S = &S_buffer[0];
        
        S[0] = static_cast<T>(0);
        
        const I step = n / thread_count;
        const I corr = n % thread_count;
        
        // We don't know the neutral element of f, so we start with a[0];
        b[0] = S[0] = a[0];
        // b[0] is already known. We have to work on the remaining array.
        
        ParallelDo(
            [=]( const I thread )
            {
                // now each thread does the scan on its chunk
                const I i_begin = step*(thread  ) + (corr*(thread  ))/thread_count + 1;
                const I i_end   = step*(thread+1) + (corr*(thread+1))/thread_count + 1;
                
                if( i_end > i_begin )
                {
                    T s_local  = a[i_begin];
                    b[i_begin] = s_local;
                    
                    for( I i = i_begin+1; i < i_end; ++i )
                    {
                        s_local = f(s_local, a[i]);
                        b[i] = s_local;
                    }
                    S[thread+1] = s_local;
                }
            },
            thread_count
        );
        
        // scan through the last results of each chuck
        {
            T s = S[0];
            for( I i = 0; i < thread_count; ++i )
            {
                s = f(s, S[i+1]);
                S[i+1] = s;
            }
        }
        
        ParallelDo(
            [=]( const I thread )
            {
                // now each thread corrects its result
                const T correction = S[thread];
                
                const I i_begin = step*(thread  ) + (corr*(thread  ))/thread_count + 1;
                const I i_end   = step*(thread+1) + (corr*(thread+1))/thread_count + 1;
                
                for( I i = i_begin; i < i_end; ++i )
                {
                    b[i] = f(b[i], correction);
                }
            },
            thread_count
        );
    }
    
    

    
    template<typename T, IntQ I>
    inline void parallel_accumulate( mptr<T> a, const I n, const I thread_count )
    {
        std::vector<T> S_buffer ( static_cast<Size_T>(thread_count+1) );
        
        mptr<T> S = &S_buffer[0];
        S[0] = 0;
        
        const I step = n / thread_count;
        const I corr = n % thread_count;
        
        ParallelDo(
            [=]( const I thread )
            {
                // now each thread does the scan on its chunk
                const I i_begin = step*(thread  ) + (corr*(thread  ))/thread_count;
                const I i_end   = step*(thread+1) + (corr*(thread+1))/thread_count;
                
                if( i_end > i_begin )
                {
                    T s_local = 0;
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        s_local += a[i];
                        a[i] = s_local;
                    }
                    S[thread+1] = a[i_end-1];
                }
                else
                {
                    S[thread+1] = 0;
                }
            },
            thread_count
        );
        
        // scan through the last results of each chunk
        {
            T s_local = 0;
            for( I thread = 0; thread < thread_count; ++thread )
            {
                s_local += S[thread+1];
                S[thread+1] = s_local;
            }
        }
        
        ParallelDo(
            [=]( const I thread )
            {
                // now each thread corrects its result
                const T correction = S[thread];
                
                const I i_begin = step*(thread  ) + (corr*(thread  ))/thread_count;
                const I i_end   = step*(thread+1) + (corr*(thread+1))/thread_count;
                
                for( I i = i_begin; i < i_end; ++i )
                {
                    a[i] += correction;
                }
            },
            thread_count
        );
    }
    
    template<Size_T N, Parallel_T parQ, typename T, IntQ Int = Size_T>
    inline void Accumulate(
        mptr<T> a, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        if constexpr ( N <= VarSize )
        {
            if constexpr (parQ == Parallel)
            {
                if( thread_count > Int(1) )
                {
                    parallel_accumulate(a,n,thread_count);
                }
                else
                {
                    for( Int i = 1; i < n; ++i ) { a[i] += a[i-Int(1)]; }
                }
            }
            else
            {
                for( Int i = 1; i < n; ++i ) { a[i] += a[i-Int(1)]; }
            }
        }
        else
        {
            (void)thread_count;
            constexpr Int N_ = static_cast<Int>(N);
            for( Int i = 1; i < N_; ++i ) { a[i] += a[i-1]; }
        }
    }
    
    template<Parallel_T parQ, typename T, IntQ Int = Size_T>
    inline void Accumulate( mptr<T> a, const Int n, const Int thread_count = 1 )
    {
        Accumulate<VarSize,parQ>(a,n,thread_count);
    }

    template<typename T, IntQ I>
    inline void parallel_accumulate(
        cptr<T> a, mptr<T> b, const I n, const I thread_count
    )
    {
        std::vector<T> S_buffer ( static_cast<Size_T>(thread_count+1) );
        mptr<T> S = &S_buffer[0];
        S[0] = 0;
        
        const I step = n / thread_count;
        const I corr = n % thread_count;
        
        ParallelDo(
            [=]( const I thread )
            {
                // now each thread does the scan on its chunk
                const I i_begin = step*(thread  ) + (corr*(thread  ))/thread_count;
                const I i_end   = step*(thread+1) + (corr*(thread+1))/thread_count;
                
                if( i_end > i_begin )
                {
                    T s_local = 0;
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        s_local += a[i];
                        b[i] = s_local;
                    }
                    S[thread+1] = b[i_end-1];
                }
                else
                {
                    S[thread+1] = static_cast<T>(0);
                }
            },
            thread_count
        );
        
        // scan through the last results of each chunk
        {
            T s_local = static_cast<T>(0);
            for( I thread = 0; thread < thread_count; ++thread )
            {
                s_local += S[thread+1];
                S[thread+1] = s_local;
            }
        }
        
        ParallelDo(
            [=]( const I thread )
            {
                // now each thread corrects its result
                const T correction = S[thread];
                
                const I i_begin = step*(thread  ) + (corr*(thread  ))/thread_count;
                const I i_end   = step*(thread+1) + (corr*(thread+1))/thread_count;
                
                for( I i = i_begin; i < i_end; ++i )
                {
                    b[i] += correction;
                }
            },
            thread_count
        );
    }
    
    template<Size_T N, Parallel_T parQ, typename T, IntQ Int = Size_T>
    inline void Accumulate(
        cptr<T> a, mptr<T> b, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        if constexpr ( N <= VarSize )
        {
            if( n <= Int(0) ) { return; }
            
            if constexpr (parQ == Parallel)
            {
                if( thread_count > Int(1) )
                {
                    parallel_accumulate(a,b,n,thread_count);
                }
                else
                {
                    b[0] = a[0];
                    for( Int i = 1; i < n; ++i ) { b[i] = a[i] + b[i-Int(1)]; }
                }
            }
            else
            {
                b[0] = a[0];
                for( Int i = 1; i < n; ++i ) { b[i] = a[i] + b[i-Int(1)]; }
            }
        }
        else
        {
            (void)thread_count;
            constexpr Int N_ = static_cast<Int>(N);
            if( N_ <= Int(0) ) { return; }
            b[0] = a[0];
            for( Int i = 1; i < N_; ++i ) { b[i] = a[i] + b[i-Int(1)]; }
        }
    }
    
    template<Parallel_T parQ, typename T, IntQ Int = Size_T>
    inline void Accumulate( cptr<T> a, mptr<T> b, const Int n, const Int thread_count = 1 )
    {
        Accumulate<VarSize,parQ>(a,b,n,thread_count);
    }
    
} // namespace Tools
