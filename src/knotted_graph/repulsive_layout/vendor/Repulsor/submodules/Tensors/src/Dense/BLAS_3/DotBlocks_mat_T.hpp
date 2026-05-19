void DotBlocks()
{
    TOOLS_PTIMER(timer,ClassName()+"::DotBlocks_mat_T");
    
    //TODO: Recursive implementation using quadtree for C.
    
    ParallelDo(
        [=,this]( const Int thread )
        {
            const Int M_blk_begin = JobPointer( M_BlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_BlockCount(), thread_count, thread + 1 );

            using a_T = mat_T<k,m,A_T>;
            using b_T = mat_T<n,k,B_T>;
            using c_T = mat_T<n,m,C_T>;

            a_T a;
            b_T b;
            c_T c;
            
            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_BlockCount(); ++N_blk )
                {
                    zerofy_buffer<m*n>( get_ptr(c) );
                    
                    for( Int K_blk = 0; K_blk < K_BlockCount(); ++K_blk )
                    {
                        copy_buffer<k*m>( AP.data(M_blk,K_blk), get_ptr(a) );
                        copy_buffer<n*k>( BP.data(N_blk,K_blk), get_ptr(b) );
                        
                        c += b * a;
                    }
                    
                    copy_buffer<n*m>( get_ptr(c), CP.data(M_blk,N_blk) );
                }
            }
        },
        thread_count
    );
}
