#pragma once

#include <bit>

namespace Tools
{
    template<typename I, typename J >
    inline constexpr void set_bit( I & x, const J pos, const bool digit )
    {
        const I mask = ~(static_cast<I>(1) << pos);
        
        x = (x & mask) | (static_cast<I>(digit) << pos);
    }
    
    template<typename I, typename J>
    inline constexpr bool get_bit( const I & x, const J pos )
    {
        const I mask = static_cast<I>( static_cast<I>(1) << pos );
        
        return ((x & mask) != I(0));
    }
    
    template<typename I, typename J >
    inline constexpr void activate_bit( I & x, const J pos )
    {
        x |= (static_cast<I>(1) << pos);
    }
    
    template<typename I, typename J >
    inline constexpr void deactivate_bit( I & x, const J pos )
    {
        x &= ~(static_cast<I>(1) << pos);
    }
    
    
    
    
    template<typename I, typename J>
    inline constexpr void set_byte( I & x, const J pos, const I b )
    {
        const I mask = ( ~static_cast<I>(0xFF) ) << (pos * 8);
        
        x = (x & mask) | (static_cast<I>(b) << (pos * 8));
    }
    
    template<typename I, typename J>
    inline constexpr I get_byte( const I & x, const J pos )
    {
        constexpr I mask = static_cast<I>(0xFF);
        
        return ( x >> (pos * 8) ) & mask;
    }
    
    template<typename UInt>
    inline constexpr int MSB( const UInt n )
    {
        static_assert(UnsignedIntQ<UInt>,"");
        
        if( n <= static_cast<UInt>(0) )
        {
            eprint("MSB called on 0.");
        }
        return std::numeric_limits<UInt>::digits - std::countl_zero(n);
    }

    
    template<typename UInt>
    inline constexpr UInt PrevPow( const UInt n )
    {
        static_assert(UnsignedIntQ<UInt>,"");
        
        if( n <= static_cast<UInt>(0) )
        {
            eprint("PrevPow called on 0.");
        }
        
        // PrevPow(0)  - Forbidden.
        // PrevPow(1)  = 1;
        
        // PrevPow(2)  = 2;
        // PrevPow(3)  = 2;
        
        // PrevPow(4)  = 4;
        // PrevPow(5)  = 4;
        // PrevPow(6)  = 4;
        // PrevPow(7 ) = 4;
        
        // PrevPow(8 ) = 8;
        // PrevPow(9 ) = 8;
        // PrevPow(10) = 8;
                
        return ( static_cast<UInt>(1) << (MSB( n ) - 1) );
    }
    
    template<typename UInt>
    inline constexpr UInt NextPow( const UInt n )
    {
        static_assert(UnsignedIntQ<UInt>,"");
        
        constexpr UInt zero = 0;
        constexpr UInt one = 1;
        
        // NextPow(0)  =  1;
        
        // NextPow(1)  =  1;
        
        // NextPow(2)  =  2;
        
        // NextPow(3)  =  4;
        // NextPow(4)  =  4;
        
        // NextPow(5)  =  8;
        // NextPow(6)  =  8;
        // NextPow(7)  =  8;
        // NextPow(8)  =  8;
        
        // NextPow(9 ) = 16;
        // NextPow(10) = 16;
                
        return (n != zero) ? ( one << ( 8 * sizeof(UInt) - std::countl_zero(n-one) ) ) : one;
    }
    
} // namespace Tools
