#pragma once

namespace Tensors
{
    template<typename T, IntQ Int>
    Int UniteSortedBuffers(
        cptr<T> u, const Int m,
        cptr<T> v, const Int n,
        mptr<T> w
    )
    {
        // Assumes that u, v are arrays of lengths m, n respectively.
        // Writes the sorted union of u and v to w and returns the length of w.
        // The caller is responsible for allocating w such that the result fits into w.
        Int i = 0;
        Int j = 0;
        Int k = 0;
        
        T u_i;
        T v_j;
        T w_k;
        
        if( i < m )
        {
            u_i = u[i];
        }
        else if( j < n )
        {
            w[0] = w_k = v[j];
            
            if( ++j < n )
            {
                v_j = v[j];
                
                goto v_remainder;
            }
            else
            {
                goto Exit;
            }
        }
        else
        {
            return 0;
        }
        
        // If we arrive here, then u_i is defined.
        if( j < n )
        {
            v_j = v[j];
        }
        else
        {
            w[0] = w_k = u_i;
            
            if( ++i < m )
            {
                u_i = u[i];
                
                goto u_remainder;
            }
            else
            {
                goto Exit;
            }
        }
        
        // If we arrive here, then both u_i and v_j are defined.
        if( u_i <= v_j )
        {
            w[0] = w_k = u_i;
            
            if( ++i < m )
            {
                u_i = u[i];
            }
            else
            {
                goto v_remainder;
            }
        }
        else
        {
            w[0] = w_k = v_j;
            
            if( ++j < n )
            {
                v_j = v[j];
            }
            else
            {
                goto u_remainder;
            }
        }
        
        // If we arrive here, then u_i, v_j, and w_k are all defined.
        
        while( true )
        {
            if( u_i <= v_j )
            {
                if( w_k != u_i ) { w[++k] = w_k = u_i; }
                
                if( ++i < m )
                {
                    u_i = u[i];
                }
                else
                {
                    goto v_remainder;
                }
            }
            else
            {
                if( w_k != v_j ) { w[++k] = w_k = v_j; }
                
                if( ++j < n )
                {
                    v_j = v[j];
                }
                else
                {
                    goto u_remainder;
                }
            }
        }
        
    u_remainder:    // treat remaining part of u
        
        while( true )
        {
            if( w_k != u_i ) { w[++k] = w_k = u_i; }
            
            if( ++i < m )
            {
                u_i = u[i];
            }
            else
            {
                goto Exit;
            }
        }
        
    v_remainder:    // treat remaining part of u
        
        while( true ) // treat remaining part of v
        {
            if( w_k != v_j ) { w[++k] = w_k = v_j; }
            
            if( ++j < n )
            {
                v_j = v[j];
            }
            else
            {
                goto Exit;
            }
        }
        
    Exit:
        
        return k+1;
    }
    
    
} // namespace Tensors
