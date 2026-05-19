template<typename A_T>
void Read_N( cptr<A_T> A, const Int ldA )
{
    //TODO: Handle boundary cases.
    
    TOOLS_PTIMER(timer,ClassName()+"::Read_N<" + TypeName<A_T> + ">");

    ParallelDo(
        [=,this]( const Int thread )
        {
            constexpr Int M_step = NotTransposedQ(op) ? m : n;
            constexpr Int N_step = NotTransposedQ(op) ? n : m;
            
            const Int M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );
            
            const Int A_step = ldA * M_step;
            
            Block_T a;
            
            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                {
                    a.template Read<op>(
                        &A[A_step * M_blk + N_step * N_blk], ldA
                    );
                    
                    a.template Write<Op::Id>( data(M_blk,N_blk) );
                }
            }
        },
        thread_count
    );
}
    
template<typename A_T>
void Read_T( cptr<A_T> A, const Int ldA )
{
    //TODO: Handle boundary cases.

    TOOLS_PTIMER(timer,ClassName()+"::Read_T<" + TypeName<A_T> + ">");
    
    ParallelDo(
        [A,ldA,this]( const Int thread )
        {
            constexpr Int M_step = NotTransposedQ(op) ? m : n;
            constexpr Int N_step = NotTransposedQ(op) ? n : m;
            
            const Int M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );
            
            Block_T a;
            
            for( Int N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
            {
                for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                {
                    a.template Read<op>(
                        &A[ldA * M_step * N_blk + N_step * M_blk], ldA
                    );
                    
                    a.Write( data(M_blk,N_blk) );
                }
            }
        },
        thread_count
    );
}
