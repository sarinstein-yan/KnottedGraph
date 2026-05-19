void DotBlocks()
{
    TOOLS_PTIMER(timer,ClassName()+"::DotBlocks_with_pointers");
    
    //TODO: Recursive implementation using quadtree for C.
        
    const Int M_blk_max = M_BlockCount();
    const Int N_blk_max = N_BlockCount();
    const Int K_blk_max = K_BlockCount();
    
    cptr<A_T> A = AP.blocks.data();
    cptr<B_T> B = BP.blocks.data();
    mptr<C_T> C = CP.blocks.data();
    
    ParallelDo(
        [=,this]( const Int thread )
        {
            const Int M_blk_begin = JobPointer( M_blk_max, thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_blk_max, thread_count, thread + 1 );

            
            using a_T = mat_T<k,m,A_T>;
            using b_T = mat_T<n,k,B_T>;
            using c_T = mat_T<n,m,C_T>;

            a_T a;
            b_T b;
            c_T c;
            
            const Int ldA = a_size * K_blk_max;
            const Int ldB = b_size * N_blk_max;
            const Int ldC = c_size * N_blk_max;
            
            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_blk_max; ++N_blk )
                {
                    zerofy_buffer<m*n>( get_ptr(c) );
                    
                    for( Int K_blk = 0; K_blk < K_blk_max; ++K_blk )
                    {
                        copy_buffer<k*m>( &A[ldA * M_blk + a_size * K_blk], get_ptr(a) );
                        copy_buffer<n*k>( &B[ldB * N_blk + b_size * K_blk], get_ptr(b) );
                        
                        c += b * a;
                    }
                    
                    copy_buffer<n*m>( get_ptr(c), &C[ldC * M_blk + c_size * N_blk] );
                }
            }
        },
        thread_count
    );
}

