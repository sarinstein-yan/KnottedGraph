#pragma once

namespace Tools
{
    // From https://stackoverflow.com/a/49658950/8248900.
    template<IntQ Int_1, IntQ Int_0>
    TOOLS_FORCE_INLINE constexpr Int_1 int_cast( const Int_0 n )
    {
        if( !std::in_range<Int_1>( n ) )
        {
            eprint(std::string("int_cast<") + TypeName<Int_1> + "," + TypeName<Int_0> + "> reports integer overflow for n = " + ToString(n) + ".");
            
            return 0;
        }
        
        return static_cast<Int_1>(n);
    }
    
    
    template<typename T>
    TOOLS_FORCE_INLINE void assert_positive( const T x )
    {
#ifdef TOOLS_DEBUG
        if constexpr ( std::numeric_limits<T>::is_signed )
        {
            if( x <= static_cast<T>(0) )
            {
                eprint(std::string("assert_positive failed in function in ") + std::string(__FILE__) + " at line "+ ToString(__LINE__)+".");
            }
        }
#else
        (void)x;
#endif
    }
    
} // namespace Tools

