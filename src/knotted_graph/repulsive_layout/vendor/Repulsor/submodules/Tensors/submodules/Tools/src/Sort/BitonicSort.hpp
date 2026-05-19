#pragma once

#include <functional>

namespace Tools
{
    template<Size_T n, Size_T vector_size, bool reverseQ, typename T, typename C>
    void BitonicMerge( mptr<T> a, C comp )
    {
        if constexpr( n > Size_T(1) )
        {
            constexpr Size_T m = n >> 1;
            
            constexpr bool lessQ = std::is_same_v<C,std::less<T>> || std::is_same_v<C,std::less_equal<T>>;
            
            constexpr bool greaterQ = std::is_same_v<C,std::greater<T>> || std::is_same_v<C,std::greater_equal<T>>;
            
            
            if constexpr ( VectorizableQ<T> && ( lessQ || greaterQ ) )
            {
                constexpr Size_T v_size = std::min( m, vector_size );
                
                constexpr Size_T v_count = m / v_size;
                
                using V_T = vec_T<v_size,T>;
                                
                V_T A;
                V_T B;
                
                copy_buffer<v_size>( &a[0], get_ptr(A) );
                copy_buffer<v_size>( &a[m], get_ptr(B) );
                
                V_T lo;
                V_T hi;
                
                for( Size_T i = 0; i < v_count; ++i )
                {
                    if constexpr ( reverseQ == lessQ )
                    {
                        lo = __builtin_elementwise_max( A[i], B[i] );
                        hi = __builtin_elementwise_min( A[i], B[i] );
                    }
                    else
                    {
                        lo = __builtin_elementwise_min( A[i], B[i] );
                        hi = __builtin_elementwise_max( A[i], B[i] );
                    }
                   
                    A[i] = lo;
                    B[i] = hi;
                }
                
                copy_buffer<v_size>( get_ptr(A), &a[0] );
                copy_buffer<v_size>( get_ptr(B), &a[m] );
            }
            else
            {
                for( Size_T i = 0; i < m; ++i )
                {
                    CompSwap<reverseQ>( a[i], a[m+i], comp );
                }
            }

            BitonicMerge<m,vector_size,reverseQ>( &a[0], comp );
            BitonicMerge<m,vector_size,reverseQ>( &a[m], comp );
        }
    }
    
    template<
        Size_T n, Size_T vector_size = 8, bool reverseQ = false,
        typename T, typename C = std::less<T>
    >
    void BitonicSort( mptr<T> a, C comp = C() )
    {
        if constexpr( n > Size_T(32) )
        {
            constexpr Size_T m = n >> 1;

            BitonicSort <m,vector_size,false   ,T,C>( &a[0], comp );
            BitonicSort <m,vector_size,true    ,T,C>( &a[m], comp );
            BitonicMerge<n,vector_size,reverseQ,T,C>( &a[0], comp );
        }
//        else if constexpr( n == 64 )
//        {
//            SortNet<64,reverseQ>()( a, comp );
//        }
        else if constexpr( n == Size_T(32) )
        {
            SortNet<32,reverseQ>()( a, comp );
        }
        else if constexpr( n == Size_T(16) )
        {
            SortNet<16,reverseQ>()( a, comp );
        }
        else if constexpr( n == Size_T(8) )
        {
            SortNet<8,reverseQ>()( a, comp );
        }
        else if constexpr( n == Size_T(4) )
        {
            SortNet<4,reverseQ>()( a, comp );
        }
        else if constexpr( n == Size_T(2) )
        {
            SortNet<2,reverseQ>()( a, comp );
        }
    }

} // namespace Tools
