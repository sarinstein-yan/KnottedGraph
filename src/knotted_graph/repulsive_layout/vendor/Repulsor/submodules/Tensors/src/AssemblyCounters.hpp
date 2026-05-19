#pragma once

namespace Tensors
{
    
//    template<IntQ LInt, IntQ Int>
//    inline void AccumulateAssemblyCounters( mref<Tensor2<LInt,Int>> counters )
//    {
//        TOOLS_PTIMER(timer, std::string( "AccumulateAssemblyCounters")
//            + "<" + TypeName<LInt>
//            + "," + TypeName<Int>
//            + ">" );
//
//        const Int thread_count = counters.Dim(0);
//
//        const Int m = counters.Dim(1);
//        
//        if( (m <= 0) || (thread_count <= 0) ) { return; }
//
//        for( Int thread = 1; thread < thread_count; ++thread )
//        {
//            counters(thread, 0) += counters(thread-1, 0);
//        }
//        for( Int i = 1; i < m; ++i )
//        {
//            counters(0, i) += counters(thread_count-1, i-1);
//
//            for( Int thread = 1; thread < thread_count; ++thread )
//            {
//                counters(thread, i) += counters(thread-1, i);
//            }
//        }
//    }


    template<Parallel_T parQ, IntQ LInt, IntQ Int>
    inline void AccumulateAssemblyCounters( mref<Tensor2<LInt,Int>> counters )
    {
        static_assert(CacheLineWidth % sizeof(LInt) == 0, "sizeof(LInt) is not a divisor of CacheLineWidth.");

        constexpr Int per_line = CacheLineWidth / sizeof(LInt);

        std::string tag = std::string("AccumulateAssemblyCounters")
            + "<" + TypeName<LInt>
            + "," + TypeName<Int>
            + "," + ToString(parQ)
            + ">";
        
        TOOLS_PTIMER(timer,tag);
        
        const Int thread_count = counters.Dim(0);
        
        const Int m = counters.Dim(1);
        
        if( (m <= 0) || (thread_count <= 0) ) { return; }
        
        if constexpr ( parQ == Sequential )
        {
            if( thread_count != 1 )
            {
                wprint(std::string("AccumulateAssemblyCounters") + ": called in sequential mode with thread_count != 1.");
            }
            for( Int thread = 1; thread < thread_count; ++thread )
            {
                counters(thread, 0) += counters(thread-1, 0);
            }
            for( Int i = 1; i < m; ++i )
            {
                counters(0, i) += counters(thread_count-1, i-1);
                
                for( Int thread = 1; thread < thread_count; ++thread )
                {
                    counters(thread, i) += counters(thread-1, i);
                }
            }
        }
        else
        {
            const Int line_count = int_cast<Int>( ( ToSize_T(m) * sizeof(LInt) + CacheLineWidth - 1 ) / CacheLineWidth );
            
            Tensor1<LInt,Int> S_buffer ( thread_count+1 );
            
            mptr<LInt> S = S_buffer.data();
            
            S[0] = LInt(0);

            const Int step = line_count / thread_count;
            const Int corr = line_count % thread_count;
            
            Do<parQ>(
                [=,&counters]( const Int thread )
                {
                    // each thread does the accumulation on its chunk independently
                    const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
                    const Int j_end   = Min(m, static_cast<Int>( (step*(thread+Int(1)) + (corr*(thread+Int(1)))/thread_count) * per_line) );
                    
                    if( j_end > j_begin )
                    {
                        for( Int i = 1; i < thread_count; ++i )
                        {
                            counters[i][j_begin] += counters[i-1][j_begin];
                        }
                        
                        for( Int j = j_begin+1; j < j_end; ++j )
                        {
                            counters[0][j] += counters[thread_count-1][j-1];
                            
                            for( Int i = 1; i < thread_count; ++i )
                            {
                                counters[i][j] += counters[i-1][j];
                            }
                        }
                        
                        S[thread+1] = counters(thread_count-1,j_end-1);
                    }
                    else
                    {
                        S[thread+1] = LInt(0);
                    }
                },
                thread_count, thread_count
            );

            // scan through the last results of each chunk
            {
                LInt s_local = LInt(0);
                for( Int i = 0; i < thread_count; ++i )
                {
                    s_local += S[i+1];
                    S[i+1] = s_local;
                }
            }

            Do<parQ>(
                [=,&counters]( const Int thread )
                {
                    // each thread adds-in its correction
                    const LInt correction = S[thread];
                    
                    const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
                    const Int j_end   = Min(m, static_cast<Int>( (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line) );

                    
                    for( Int i = 0; i < thread_count; ++i )
                    {
                        mptr<LInt> c_i = counters.data(i);
                        
                        for( Int j = j_begin; j < j_end; ++j )
                        {
                            c_i[j] += correction;
                        }
                    }
                },
                thread_count, thread_count
            );
        }
    }




    template<IntQ LInt, IntQ Int, Parallel_T parQ>
    inline Tensor2<LInt,Int> AssemblyCounters(
        const  Int * const * const idx,
        const  Int * const * const jdx,
        const LInt * entry_counts,
        const  Int list_count,
        const  Int m,
        const  int symmetrize = 0
    )
    {
        std::string tag = std::string( "AssemblyCounters") 
            + "<" + TypeName<LInt>
            + "," + TypeName<Int>
            + ">";
        
        TOOLS_PTIMER(timer,tag);
        
        Tensor2<LInt,Int> counters ( list_count, m, LInt(0) );

        // https://en.wikipedia.org/wiki/Counting_sort
        // using parallel count sort to sort the cluster (i,j)-pairs according to i.
        // storing counters of each i-index in thread-interleaved format
        // TODO: Improve data layout (transpose counts).
        
        Do<parQ>(
            [=,&counters]( const Int thread )
            {
                cptr<Int> thread_idx = idx[thread];
                cptr<Int> thread_jdx = jdx[thread];
                
                const LInt entry_count = entry_counts[thread];
                
                mptr<LInt> c = counters.data(thread);
                
                if( symmetrize!=0 )
                {
                    for( LInt k = 0; k < entry_count; ++k )
                    {
                        const Int i = thread_idx[k];
                        const Int j = thread_jdx[k];
                        
                        c[i] ++;
                        c[j] += static_cast<Int>(i != j);
                    }
                }
                else
                {
                    for( LInt k = 0; k < entry_count; ++k )
                    {
                        const Int i = thread_idx[k];
                        
                        ++c[i];
                    }
                }
            },
            list_count, list_count
        );
        
    //        print(counters.ToString());
        AccumulateAssemblyCounters<parQ>(counters);
    //        print(counters.ToString());
        
        return counters;
    }
    
}
