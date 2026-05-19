#pragma once

namespace Tools
{
    
    template<typename T, IntQ Int>
    void reverse_buffer(
        mptr<T> x, const Int n
    )
    {
        T buffer;
        
        for( Int i = 0; i < n/2; ++i )
        {
            buffer   = std::move(x[i]);
            x[i]     = std::move(x[n-i-1]);
            x[n-i-1] = std::move(buffer);
        }
//        std::reverse(&x[0],&x[n]);
    }
    
    template<Side dir, typename T, IntQ Int>
    void rotate_buffer(
        mptr<T> x, const Int shift, const Int n
    )
    {
        if( n <= Int(1) )
        {
            return;
        }
        
        Int s = (dir == Side::Left) ? (shift % n) : (-shift % n);
        
        if( s < Int(0) )
        {
            s += n;
        }
        
        if( s == Int(0) )
        {
            return;
        }
        
        const Int k = n - s;
        
        reverse_buffer(&x[0],s);
        reverse_buffer(&x[s],k);
        
        reverse_buffer(&x[0],n);
    }
    
} // namespace Tools


