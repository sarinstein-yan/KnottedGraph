#pragma once

namespace Tensors
{
    template<
        Size_T N = VarSize, typename C = std::less<>,
        typename T, IntQ Int
    >
    void Ordering( cptr<T> key, mptr<Int> perm, Int n, C comp = C() )
    {
        if( n <= Scalar::Zero<Int> )
        {
            return;
        }
        
        if( n == Scalar::One<Int> )
        {
            perm[0] = 0;
            return;
        }
        
        iota_buffer<N,Sequential>( perm, static_cast<Size_T>(n) );
        
        Sort( perm, &perm[n],
            [=]( const Int i, const Int j ) -> bool
            {
                return comp( key[i], key[j] );
            }
        );
    }
    
    template<
        Size_T N = VarSize, typename C = std::less<>,
        typename T, IntQ Int
    >
    void Ordering( cptr<T> key, Int stride, mptr<Int> perm, Int n, C comp = C() )
    {
        if( n <= Scalar::Zero<Int> )
        {
            return;
        }
        
        if( n == Scalar::One<Int> )
        {
            perm[0] = 0;
            return;
        }
        
        iota_buffer<N,Sequential>( perm, n );
        
        Sort( perm, &perm[n],
            [=]( const Int i, const Int j ) -> bool
            {
                return comp( key[stride * i], key[stride * j] );
            }
        );
    }

}


