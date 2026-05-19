#pragma once

#include "Ordering.hpp"

namespace Tensors
{
    template<typename Key_T, typename V_0, typename V_1, IntQ Int,
        Size_T N = VarSize, typename C = std::less<Key_T>
    >
    class ThreeArraySort
    {
    private:
        
        Tensor1<Int,  Int> perm;
        Tensor1<Key_T,Int> keys;
        Tensor1<V_0,Int>   v_0;
        Tensor1<V_1,Int>   v_1;
        C comp;
        
    public:
        
        ThreeArraySort( Int n = N, C comp_ = C() )
        :   perm ( n )
        ,   keys ( n )
        ,   v_0  ( n )
        ,   v_1  ( n )
        ,   comp ( comp_ )
        {}
        
        void operator()( mptr<Key_T> keys_, mptr<V_0> v_0_, mptr<V_1> v_1_, const Int n = N )
        {
            // Sort the elements in array `v_0` and `v_1` according to the corresp. elements in the array `keys`.
            
            if( n <= Scalar::One<Int> )
            {
                return;
            }
            
            if( n > perm.Size() )
            {
                perm = Tensor1<Int,  Int> ( n );
                keys = Tensor1<Key_T,Int> ( n );
                v_0  = Tensor1<V_0,Int> ( n );
                v_1  = Tensor1<V_1,Int> ( n );
            }
            
            move_buffer<N>( keys_, keys.data(), n );

            Ordering<N>( keys.data(), perm.data(), n, comp );
            
            move_buffer<N>( v_0_, v_0.data(), n );
            move_buffer<N>( v_1_, v_1.data(), n );
                
            {
                for( Int i = 0; i < n; ++i )
                {
                    const Int perm_i = perm[i];
                    
                    keys_[i] = std::move(keys[perm_i]);
                    v_0_ [i] = std::move(v_0 [perm_i]);
                    v_1_ [i] = std::move(v_1 [perm_i]);
                }
            }
            
        }
        
    }; // ThreeArraySort
    
}


