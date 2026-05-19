#pragma once

#include "Ordering.hpp"

namespace Tensors
{
    template<typename Key_T, typename Val_T, IntQ Int,
        Size_T N = VarSize, typename C = std::less<Key_T>
    >
    class TwoArraySort
    {
    private:
        
        Tensor1<Int,  Int> perm;
        Tensor1<Key_T,Int> keys;
        Tensor1<Val_T,Int> vals;
        C comp;
        
    public:
        
        TwoArraySort( Int n = N, C comp_ = C() )
        :   perm ( n )
        ,   keys ( n )
        ,   vals ( n )
        ,   comp ( comp_ )
        {}
        
        void operator()( mptr<Key_T> keys_, mptr<Val_T> vals_, const Int n = N )
        {
            // Sort the elements in array `vals` according to the corresp. elements in the array `keys`.
            
            if( n <= Scalar::One<Int> )
            {
                return;
            }
            
            if( n > perm.Size() )
            {
                perm = Tensor1<Int,  Int> ( n );
                keys = Tensor1<Key_T,Int> ( n );
                vals = Tensor1<Val_T,Int> ( n );
            }
            
            move_buffer<N>( keys_, keys.data(), n );

            Ordering<N>( keys.data(), perm.data(), n, comp );
            
            move_buffer<N>( vals_, vals.data(), n);
                
            {
                for( Int i = 0; i < n; ++i )
                {
                    const Int perm_i = perm[i];
                    
                    keys_[i] = std::move(keys[perm_i]);
                    vals_[i] = std::move(vals[perm_i]);
                }
            }
            
        }
        
    }; // TwoArraySort
    
}

