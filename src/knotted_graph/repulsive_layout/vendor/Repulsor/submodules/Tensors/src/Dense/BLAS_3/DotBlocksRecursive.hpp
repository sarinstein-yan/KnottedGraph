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
    
    constexpr Int M_blk_count_min =  4 * 2 * 2 * 2;
    constexpr Int N_blk_count_min =  8 * 2 * 2 * 2;
    constexpr Int K_blk_count_min = 16 * 2 * 2 * 2 ;
    
    constexpr Int M_over_N_min = M_blk_count_min / N_blk_count_min;
    constexpr Int M_over_K_min = K_blk_count_min / M_blk_count_min;
    constexpr Int N_over_K_min = K_blk_count_min / N_blk_count_min;
    
    if( (M_blk_count >= M_over_N_min * N_blk_count) && (M_blk_count >= M_over_K_min*K_blk_count) && (M_blk_count > M_blk_count_min) )
    {
        const Int M_blk_mid = M_blk_begin + (M_blk_count/2);
        
        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_begin, N_blk_end, K_blk_begin, K_blk_end );
        dotBlocksRecursive( M_blk_mid  , M_blk_end, N_blk_begin, N_blk_end, K_blk_begin, K_blk_end );
    }
    else if( (N_blk_count >= N_over_K_min*K_blk_count) && (N_blk_count > N_blk_count_min) )
    {
        const Int N_blk_mid = N_blk_begin + (N_blk_count/2);
        
        dotBlocksRecursive( M_blk_begin, M_blk_end, N_blk_begin, N_blk_mid, K_blk_begin, K_blk_end );
        dotBlocksRecursive( M_blk_begin, M_blk_end, N_blk_mid  , N_blk_end, K_blk_begin, K_blk_end );
    }
    else if( (K_blk_count > K_blk_count_min) )
    {
        const Int K_blk_mid = K_blk_begin + (K_blk_count/2);
        
        dotBlocksRecursive( M_blk_begin, M_blk_end, N_blk_begin, N_blk_end, K_blk_begin, K_blk_mid );
        dotBlocksRecursive( M_blk_begin, M_blk_end, N_blk_begin, N_blk_end, K_blk_mid,   K_blk_end );
    }
//    else if( (M_blk_count > Int(1)) && (N_blk_count > Int(1)) && (K_blk_count > Int(1)) )
//    {
//        const Int M_blk_mid = M_blk_begin + (M_blk_count/2);
//        const Int N_blk_mid = N_blk_begin + (N_blk_count/2);
//        const Int K_blk_mid = K_blk_begin + (K_blk_count/2);
//        
//        
//        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_begin, N_blk_mid, K_blk_begin, K_blk_mid );
//        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_begin, N_blk_mid, K_blk_mid,   K_blk_end );
//        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_mid,   N_blk_end, K_blk_begin, K_blk_mid );
//        dotBlocksRecursive( M_blk_begin, M_blk_mid, N_blk_mid,   N_blk_end, K_blk_mid,   K_blk_end );
//        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_begin, N_blk_mid, K_blk_begin, K_blk_mid );
//        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_begin, N_blk_mid, K_blk_mid,   K_blk_end );
//        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_mid,   N_blk_end, K_blk_begin, K_blk_mid );
//        dotBlocksRecursive( M_blk_mid,   M_blk_end, N_blk_mid,   N_blk_end, K_blk_mid,   K_blk_end );
//    }
    else
    {
        for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
        {
            for( Int N_blk = N_blk_begin; N_blk < N_blk_end; ++N_blk )
            {
//                for( Int K_blk = K_blk_begin; K_blk < K_blk_end; ++K_blk )
//                {
//                    if constexpr( mat_enabledQ && (SameQ<A_T,double> || SameQ<A_T,float>) && (SameQ<B_T,double> || SameQ<B_T,float>) && (SameQ<C_T,double> || SameQ<C_T,float>) )
//                    {
//                        Tiny::fixed_dot_mm_clang<m,n,k,AddTo>(
//                            AP.data(M_blk,K_blk), BP.data(N_blk,K_blk), CP.data(M_blk,N_blk)
//                        );
//                    }
//                    else
//                    {
//                        Tiny::fixed_dot_mm_vec<m,n,k,AddTo>(
//                            AP.data(M_blk,K_blk), BP.data(N_blk,K_blk), CP.data(M_blk,N_blk)
//                        );
//                    }
//                }
                
                C_Block_T c ( C_T(0) );
                
                for( Int K_blk = K_blk_begin; K_blk < K_blk_end; ++K_blk )
                {
                    A_Block_T a ( AP.data(M_blk,K_blk) );
                    B_Block_T b ( BP.data(N_blk,K_blk) );

                    Dot<AddTo>( a, b, c );
                }
                
                constexpr C_T one = 1;
                
                c.template Write<Scalar::Flag::Plus,Scalar::Flag::Plus>(
                    one, one, CP.data(M_blk,N_blk)
                );
            }
        }
    }
        
}

