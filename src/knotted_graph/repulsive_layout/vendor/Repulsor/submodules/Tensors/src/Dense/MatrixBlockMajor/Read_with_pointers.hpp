template<typename A_T>
void Read_N( cptr<A_T> A, const Int ldA )
{
    //TODO: Handle boundary cases.
    
    TOOLS_PTIMER(timer,ClassName()+"::Read_N_with_pointers<" + TypeName<A_T> + ">");

    constexpr Int M_step = NotTransposedQ(op) ? m : n;
    constexpr Int N_step = NotTransposedQ(op) ? n : m;
    
    const Int M_blk_max = M_BlockCount();
    const Int N_blk_max = N_BlockCount();
    
    mptr<Scal> blocks_ = blocks.data();
    
    ParallelDo(
        [=,this]( const Int thread )
        {
            const Int M_blk_begin = JobPointer( M_blk_max, thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_blk_max, thread_count, thread + 1 );
            
            const Int ld = block_size * N_blk_max;
            
            const Int A_step = ldA * M_step;
            
            Block_T a;
            
            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_blk_max; ++N_blk )
                {
                    a.template Read<op>(
                        &A[A_step * M_blk + N_step * N_blk], ldA
                    );
                    
                    copy_buffer<block_size>( a.data(), &blocks_[ld * M_blk + block_size * N_blk] );
                    
//                            blocks[M_blk][N_blk].template Read<op>(
//                                &A[ldA * M_step * M_blk + N_step * N_blk], ldA
//                            );
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

    TOOLS_PTIMER(timer,ClassName()+"::Read_T_with_pointers<" + TypeName<A_T> + ">");
    
    ParallelDo(
        [A,ldA,this]( const Int thread )
        {
            constexpr Int M_step = NotTransposedQ(op) ? m : n;
            constexpr Int N_step = NotTransposedQ(op) ? n : m;
            
            const Int M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );
            
            Block_T a;
            
//            mptr<Scal> blocks_ = blocks.data();
            
//                    const Int ld = n * N_BlockCount();
            
            const Int N_blk_max = N_FullBlockCount();
            
            for( Int N_blk = 0; N_blk < N_blk_max; ++N_blk )
            {
                for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                {
                    a.template Read<op>(
                        &A[ldA * M_step * N_blk + N_step * M_blk], ldA
                    );
                    
                    a.Write( data(M_blk,N_blk) );
                    
//                            copy_buffer<m*n>( a.data(), &blocks_[ld * M_blk + n * N_blk] );
                    
//                            blocks[M_blk][N_blk].template Read<op>(
//                                &A[ldA * M_step * N_blk + N_step * M_blk], ldA
//                            );
                }
            }
        },
        thread_count
    );
}

