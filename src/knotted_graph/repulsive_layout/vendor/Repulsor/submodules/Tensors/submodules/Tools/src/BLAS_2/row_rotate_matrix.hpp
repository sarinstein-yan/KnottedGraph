#pragma once

namespace Tools
{
    template<
        Size_T M = VarSize, Size_T N = VarSize,
        typename T, typename Fun, IntQ Int = Size_T
    >
    void row_reverse_matrix(
        mptr<T> A,
        const Int ldA,
        mptr<T> scratch,
        const Int m,
        const Int n,
        Fun && fun
    )
    {
        if constexpr ( M > VarSize )
        {
            const Int half = M/2;
            
            for( Int i = 0; i < half; ++i )
            {
                move_buffer<N>( &A[ldA * i], scratch, n );
                fun( &A[ldA *(m-i-1)], &A[ldA * i]      );
                fun( scratch         , &A[ldA *(m-i-1)] );
            }
            
            if( M % 2 )
            {
                move_buffer<N>( &A[ldA * half], scratch, n );
                fun( scratch, &A[ldA * half] );
            }
        }
        else
        {
            const Int half = m/2;
            
            for( Int i = 0; i < half; ++i )
            {
                move_buffer<N>( &A[ldA * i], scratch, n );
                fun( &A[ldA *(m-i-1)], &A[ldA * i]      );
                fun( scratch         , &A[ldA *(m-i-1)] );
            }
            
            if( m % 2 )
            {
                move_buffer<N>( &A[ldA * half], scratch, n );
                fun( scratch, &A[ldA * half] );
            }
        }
    }
    
    template<
        Size_T M = VarSize, Size_T N = VarSize,
        typename T, IntQ Int = Size_T
    >
    void row_reverse_matrix(
        mptr<T> A,
        const Int ldA,
        const Int m = static_cast<Int>(M),
        const Int n = static_cast<Int>(N)
    )
    {
        // rotate_buffer(A,shift * n, m * n) would also work, but this does it i chunks.
        T * scratch = nullptr;
        
        auto fun = [n]( cptr<T> from, mptr<T> to )
        {
            move_buffer<N>( from, to, n );
        };
        
        safe_alloc( scratch, n );
        
        row_reverse_matrix<M,N>( A, ldA, scratch, m, n, std::move(fun) );
        
        safe_free( scratch );
    }
    
    
    template<
        Size_T M = VarSize, Size_T N = VarSize, Side dir = Side::Left,
        typename T, typename PreScan, typename PostScan, IntQ Int = Size_T
    >
    void row_rotate_matrix(
        mptr<T> A,
        const Int ldA,
        const Int shift,
        const Int m,
        const Int n,
        PreScan  && pre_scan,
        PostScan && post_scan
    )
    {
        if( m <= Int(0) ) { return; }
        
        Int s = (dir == Side::Left) ? (shift % m) : (-shift % m);
        
        if( s < Int(0) )
        {
            s += m;
        }
  
        // We must not abort here because pre_scan and post_scan need to be applied.
//        if( s == Int(0) ) { return; }
        
        const Int k = m - s;
        
        T * scratch = nullptr;
        safe_alloc( scratch, ToSize_T(n) );
        
        // This does not really work with fixed row count as s is not a compile-time constant.
        row_reverse_matrix<VarSize,N>( &A[ldA * 0], ldA, scratch, s, n, pre_scan  );
        row_reverse_matrix<VarSize,N>( &A[ldA * s], ldA, scratch, k, n, pre_scan  );
        row_reverse_matrix<M      ,N>( &A[ldA * 0], ldA, scratch, m, n, post_scan );
            
        safe_free( scratch );
    }
    
    
    template<
        Size_T M = VarSize, Size_T N = VarSize, Side dir = Side::Left,
        typename T, IntQ Int = Size_T
    >
    void row_rotate_matrix(
        mptr<T> A,
        const Int ldA,
        const Int shift,
        const Int m = static_cast<Int>(M),
        const Int n = static_cast<Int>(N)
    )
    {
        if( (shift % m) == Int(0) ) { return; }
        
        auto fun = [n]( cptr<T> from, mptr<T> to )
        {
            move_buffer<N>( from, to, n );
        };
        
        row_rotate_matrix<M,N,dir>( A, ldA, shift, m, n, std::move(fun), std::move(fun) );
    }
    
} // namespace Tools
