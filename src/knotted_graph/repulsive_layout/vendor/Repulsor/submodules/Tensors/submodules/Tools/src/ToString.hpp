#pragma once

namespace Tools
{
#ifdef TOOLS_NO_STDFORMAT
#include "fmt/format.h"
    using fmt::format;
#else
    using std::format;
#endif
    
    
    [[nodiscard]] std::string ToString( const std::string & s )
    {
        return s;
    }
    
    [[nodiscard]] std::string ToString( const char * ptr )
    {
        return std::string(ptr);
    }
    
    [[nodiscard]] std::string ToString( char * ptr )
    {
        return std::string(ptr);
    }
    
    [[nodiscard]] std::string ToString( const double & value )
    {
        return format("{: .17g}",value);
    }
    
    [[nodiscard]] std::string ToString( const float & value )
    {
        return format("{: .8g}",value);
    }
    
    [[nodiscard]] std::string ToString( const std::complex<double> & z )
    {
        return format("{: .17g} {:+.17g}I",std::real(z),std::imag(z));
    }
                           
    [[nodiscard]] std::string ToString( const std::complex<float> & z )
    {
        return format("{: .8g} {:+.8g}I",std::real(z),std::imag(z));
    }
    
    template<FloatQ T>
    [[nodiscard]] std::string ToMathematicaString( T x )
    {
        std::string s = ToString(x);
        
        Size_T p = s.find("e");
        
        if(p != std::string::npos)
        {
            s.replace(p, 1, "*^");
        }
        return s;
    }
    
    template<FloatQ T>
    std::ostream & operator<<( std::ostream & sout, const std::complex<T> & z )
    {
        return sout << ToString(z);
    }

    template<typename T>
    [[nodiscard]] std::string ToString( T * ptr )
    {
        return format( "{:p}", (void *)ptr );
    }
    
    template<typename T>
    [[nodiscard]] std::string ToString( T const * ptr )
    {
        return format( "{:p}", (void const *)ptr );
    }

    template<IntQ T>
    [[nodiscard]] std::string ToString( const T & value )
    {
        return format("{:d}",value);
    }
    
    
//    template<typename T>
//    constexpr bool EnumClassQ = std::is_enum<T>::value && !std::is_convertible<T,int>::value;
    
    template<typename T>
    concept EnumClass = std::is_enum<T>::value && !std::is_convertible<T,int>::value;
    
    
    template <EnumClass T>
    [[nodiscard]] std::string ToString( const T & value )
    {
        return ToString( ToUnderlying(value) );
    }
    
    template <typename S, typename T>
    [[nodiscard]] std::string ToString( const std::pair<S,T> & p )
    {
        std::string s ("{ ");
        s += ToString(p.first);
        s += ", ";
        s += ToString(p.second);
        s += " }";
        return s;
    }

    
//    template<typename T, NonIntQ F>
//    [[nodiscard]] std::string ToString(
//        cref<std::vector<T>> v,
//        F && fun,
//        std::string line_prefix
//    )
//    {
//        const Size_T dim = v.size();
//        return ArrayToString( &v[0], &dim, Size_T(1), fun, line_prefix );
//    }
//    
//    template<typename T>
//    [[nodiscard]] std::string ToString(
//        const std::vector<T> & v,
//        std::string line_prefix = ""
//    )
//    {
//        return ToString( v, []( cref<T> a ){ return ToString(a); }, line_prefix );
//    }
    
//    template<typename T, Size_T N, NonIntQ F>
//    [[nodiscard]] std::string ToString(
//        const std::array<T,N> & v,
//        F && fun,
//        std::string line_prefix = ""
//    )
//    {
//        const Size_T dim = N;
//        return ArrayToString( &v[0], &dim, Size_T(1), fun, line_prefix );
//    }
//    
//    template<typename T, Size_T N>
//    [[nodiscard]] std::string ToString(
//        const std::array<T,N> & v,
//        std::string line_prefix = ""
//    )
//    {
//        return ToString( v, []( cref<T> a ){ return ToString(a); }, line_prefix );
//    }
    
    template<typename T>
    [[nodiscard]] std::string StringWithLeadingZeroes(
        const T val, const int width
    )
    {
        std::stringstream s;
        s << std::setw(width) << std::setfill('0') << val;
        return s.str();
    }

    [[nodiscard]] std::string BoolString( const bool b )
    {
        return b ? "True" : "False";
    }
    
} // namespace Tools
