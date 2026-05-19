#pragma once

namespace Tensors
{
    
    template<IntQ Int>
    static constexpr Int isqrt( const Int k )
    {
        return static_cast<Int>( std::floor(cSqrt(static_cast<double>(k))) );
    }
    
    // Indexing into upper triangle _with_ diagonal
    
    template<Size_T n, IntQ Int>
    static constexpr Int tri_i( const Int k )
    {
        // https://stackoverflow.com/a/244550/8248900
     
        Int kk = n * (n + 1) / 2 - 1 - k;
        Int K = ( isqrt( 8 * kk + 1 ) - 1 ) / 2;
        return n-1-K;
    }
    
    template<Size_T n, IntQ Int>
    static constexpr Int tri_j( const Int k )
    {
        // https://stackoverflow.com/a/244550/8248900
        
        Int kk = (n * (n + 1)) / 2 - 1 - k;
        Int K = ( isqrt( 8 * kk + 1 ) - 1 ) / 2;
        return n - 1 - kk + (K * (K + 1)) /2;

    }
    
    template<Size_T n, IntQ Int>
    static constexpr Int lin_k( const Int i, const Int j )
    {
        // https://stackoverflow.com/a/244550/8248900
        
        return ( i <= j ) ?
            j + ( i * (2 * n - 1 - i ) ) / 2
            :
            i + ( j * (2 * n - 1 - j ) ) / 2;
    }
    
    
    
    // Indexing into upper triangle _without_ diagonal
    
    template<Size_T n, IntQ Int>
    static constexpr Int u_tri_i( const Int k )
    {
        // https://stackoverflow.com/a/244550/8248900
        return n - 2 - static_cast<Int>(std::floor(cSqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5));
    }
    
    template<Size_T n, IntQ Int>
    static constexpr Int u_tri_j( const Int k )
    {
        // https://stackoverflow.com/a/244550/8248900
        const Int i = u_tri_i<n>(k);
        
        return k + i + 1 - (n * (n-1)) / 2 + (n-i)*((n-i)-1)/2;

    }
    
    
    template<Size_T n, IntQ Int>
    static constexpr Int u_lin_k( const Int i, const Int j )
    {
        // https://stackoverflow.com/a/244550/8248900
        
        return ( i <= j ) ?
            (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
            :
            (n*(n-1)/2) - (n-j)*((n-i)-1)/2 + i - j - 1;
    }

    
} // namespace Tensors
