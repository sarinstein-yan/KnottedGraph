void DotBlocksRecursive()
{
    dotBlocksRecursive( 0, M_BlockCount(), 0, N_BlockCount(), 0, K_BlockCount() );
}

void dotBlocksRecursive(
    const Int M_blk_begin, const Int M_blk_end,
    const Int N_blk_begin, const Int N_blk_end,
    const Int K_blk_begin, const Int K_blk_end
    
)
{
    const Int M_blk_count = M_blk_end - M_blk_begin;
    const Int N_blk_count = N_blk_end - N_blk_begin;
    const Int K_blk_count = K_blk_end - K_blk_begin;
    
    if( (M_blk_count > Int(2)) && (N_blk_count > Int(2)) && (K_blk_count > Int(32)) )
    {
        const Int M_blk_mid = M_blk_begin + (M_blk_count/2);
        const Int N_blk_mid = N_blk_begin + (N_blk_count/2);
        const Int K_blk_mid = K_blk_begin + (K_blk_count/2);
        
        
        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_begin, N_blk_mid, K_blk_begin, K_blk_mid );
        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_begin, N_blk_mid, K_blk_mid,   K_blk_end );
        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_mid,   N_blk_end, K_blk_begin, K_blk_mid );
        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_mid,   N_blk_end, K_blk_mid,   K_blk_end );
        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_begin, N_blk_mid, K_blk_begin, K_blk_mid );
        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_begin, N_blk_mid, K_blk_mid,   K_blk_end );
        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_mid,   N_blk_end, K_blk_begin, K_blk_mid );
        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_mid,   N_blk_end, K_blk_mid,   K_blk_end );
    }
    else
    {
        constexpr C_T one = 1;
        
        A_Block_T a;
        B_Block_T b;
        C_Block_T c;
        
        for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
        {
            for( Int N_blk = N_blk_begin; N_blk < N_blk_end; ++N_blk )
            {
                c.SetZero();
                
                for( Int K_blk = K_blk_begin; K_blk < K_blk_end; ++K_blk )
                {
                    a.Read( AP.data(M_blk,K_blk) );
                    b.Read( BP.data(N_blk,K_blk) );

                    Dot<AddTo>( a, b, c );
                }
                
                c.template Write<Scalar::Flag::Plus,Scalar::Flag::Plus>(
                    one, one, CP.data(M_blk,N_blk)
                );
            }
        }
    }
        
}


