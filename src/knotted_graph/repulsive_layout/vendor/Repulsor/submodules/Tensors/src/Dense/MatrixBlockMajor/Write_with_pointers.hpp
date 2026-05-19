template<
    Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
    typename alpha_T, typename beta_T, typename C_T
>
void Write_N(
        const alpha_T alpha, const beta_T beta,
        mptr<C_T> C, const Size_T ldC
) const
{
    TOOLS_PTIMER(timer,ClassName()+"::Write_N_with_pointers<" + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + ">");

    //TODO: Handle boundary cases.
    
    ParallelDo(
        [alpha,beta,C,ldC,this]( const Size_T thread )
        {
            constexpr Size_T M_step = NotTransposedQ(op) ? m : n;
            constexpr Size_T N_step = NotTransposedQ(op) ? n : m;
            
            const Size_T M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Size_T M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

            Block_T c;
            
            for( Size_T M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Size_T N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                {
//                            blocks[M_blk][N_blk].template Write<alpha_flag,beta_flag,op,Op::Id>(
//                                alpha, beta, &C[ldC * M_step * M_blk + N_step * N_blk], ldC
//                            );
                    
                    c.Read( data(M_blk,N_blk) );
                    
                    c.template Write<alpha_flag,beta_flag,op,Op::Id>(
                        alpha, beta, &C[ldC * M_step * M_blk + N_step * N_blk], ldC
                    );
                }
            }
        },
        thread_count
    );
}
    
template<
    Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
    typename alpha_T, typename beta_T, typename C_T
>
void Write_T(
        const alpha_T alpha, const beta_T beta,
        mptr<C_T> C, const Size_T ldC
) const
{
    TOOLS_PTIMER(timer,ClassName()+"::Write_T_with_pointers<" + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + ">");
    
    //TODO: Handle boundary cases.
    
    //TODO: Test this code branch!

    ParallelDo(
        [=,this]( const Size_T thread )
        {
            constexpr Size_T M_step = NotTransposedQ(op) ? m : n;
            constexpr Size_T N_step = NotTransposedQ(op) ? n : m;
            
            const Size_T M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Size_T M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

            Block_T c;
            
            for( Size_T N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
            {
                for( Size_T M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                {
//                            blocks[M_blk][N_blk].template Write<alpha_flag,beta_flag,op,Op::Id>(
//                                &C[ldC * M_step * N_blk + N_step * M_blk], ldC
//                            );
                    
                    c.Read( data(M_blk,N_blk) );
                    
                    c.template Write<alpha_flag,beta_flag,op,Op::Id>(
                        alpha, beta, &C[ldC * M_step * N_blk + N_step * M_blk], ldC
                    );
                }
            }
        },
        thread_count
    );
}
