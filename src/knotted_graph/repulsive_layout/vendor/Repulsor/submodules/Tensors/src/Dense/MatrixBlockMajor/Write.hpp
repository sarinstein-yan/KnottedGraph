template<
    Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
    typename alpha_T, typename beta_T, typename C_T
>
void Write_N(
        const alpha_T alpha, const beta_T beta,
        mptr<C_T> C, const Int ldC
) const
{
    TOOLS_PTIMER(timer,ClassName()+"::Write_N<" + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + ">");

    //TODO: Handle boundary cases.
    
    ParallelDo(
        [alpha,beta,C,ldC,this]( const Int thread )
        {
            constexpr Int M_step = NotTransposedQ(op) ? m : n;
            constexpr Int N_step = NotTransposedQ(op) ? n : m;
            
            const Int M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

            Block_T c;
            
            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                {
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
        mptr<C_T> C, const Int ldC
) const
{
    TOOLS_PTIMER(timer,ClassName()+"::Write_T<" + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + ">");
    
    //TODO: Handle boundary cases.
    
    //TODO: Test this code branch!

    ParallelDo(
        [=,this]( const Int thread )
        {
            constexpr Int M_step = NotTransposedQ(op) ? m : n;
            constexpr Int N_step = NotTransposedQ(op) ? n : m;
            
            const Int M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

            Block_T c;
            
            for( Int N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
            {
                for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                {
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

