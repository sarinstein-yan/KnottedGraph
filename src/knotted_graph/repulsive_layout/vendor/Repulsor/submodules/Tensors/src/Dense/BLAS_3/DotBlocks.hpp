void DotBlocks()
{
    TOOLS_PTIMER(timer,ClassName()+"::DotBlocks");
    
    ParallelDo(
        [=,this]( const Int thread )
        {
            const Int M_blk_begin = JobPointer( M_BlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_BlockCount(), thread_count, thread + 1 );

            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_BlockCount(); ++N_blk )
                {
//                    C_Block_T c ( C_T(0) );
                    
                    mptr<C_T> C_ptr = CP.data(M_blk,N_blk);

                    {
                        const Int K_blk = 0;

                        Tensors::Tiny::fixed_dot_mm<m,n,k,Overwrite>(
                            AP.data(M_blk,K_blk), BP.data(N_blk,K_blk), C_ptr
                        );
                    }
                    
                    for( Int K_blk = 1; K_blk < K_BlockCount(); ++K_blk )
                    {
//                        copy_buffer<m*k>( &AP_[ldAP * M_blk + a_size * K_blk], a.data() );
//                        copy_buffer<k*n>( &BP_[ldBP * N_blk + b_size * K_blk], b.data() );
                        
//                        A_Block_T a ( AP.data(M_blk,K_blk) );
//                        B_Block_T b ( BP.data(N_blk,K_blk) );
//
//                        Dot<AddTo>( a, b, c );
                        
                        Tensors::Tiny::fixed_dot_mm<m,n,k,AddTo>(
                            AP.data(M_blk,K_blk), BP.data(N_blk,K_blk), C_ptr
                        );
                    }

//                    c.Write( CP.data(M_blk,N_blk) );
                }
            }
        },
        thread_count
    );
}
