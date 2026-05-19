#pragma once

#include <stdint.h>
#include <string>

namespace Tools
{
    // Taken from https://stackoverflow.com/a/4485051/8248900
    
    
    // default implementation
    template<typename T>
    constexpr const char * TypeName = "UnknownType";
    
    template<> constexpr const char * TypeName<Real32>  = "R32";
    
    template<> constexpr const char * TypeName<Real64>  = "R64";
    
    template<> constexpr const char * TypeName<Complex32>  = "C32";
    
    template<> constexpr const char * TypeName<Complex64>  = "C64";
    
    template<> constexpr const char * TypeName<std::string>  = "String";
    
    template<typename S, typename T> const char * TypeName<std::pair<S,T>> = "pair<?,?>";
    
    
    template<typename T> requires std::is_integral_v<T>
    constexpr const char * TypeName<T> = Scalar::IntegralTypeTraits<T>::type_name;
    
    // default implementation
    template<typename T>
    constexpr const char * FullTypeName = "UnknownType";
    
    template<> constexpr const char * FullTypeName<Real32>  = "float";
    
    template<> constexpr const char * FullTypeName<Real64>  = "double";
    
    template<> constexpr const char * FullTypeName<Complex32>  = "std::complex<float>";
    
    template<> constexpr const char * FullTypeName<Complex64>  = "std::complex<double>";
    
    template<> constexpr const char * FullTypeName<std::string>  = "std::string";
    
    template<typename S, typename T>
    constexpr const char * FullTypeName<std::pair<S,T>> = "pair<?,?>";
    
    template<typename T> requires std::is_integral_v<T>
    constexpr const char * FullTypeName<T> = Scalar::IntegralTypeTraits<T>::full_name;

    [[maybe_unused]] std::string IntegerInfo()
    {
        std::stringstream s ;

        const int  w0 = 22 + 1;
        const int  w1 = 11;
        const int  w2 = 22;
        const int  w3 = 5;
        const int  w4 = 5;
        
        
        auto line = [=,&s]<typename T>( const char * name )
        {
            s << std::right << std::setw(w0) << name << " | "
              << std::right << std::setw(w1) << TypeName<T> << " | "
              << std::right << std::setw(w2) << FullTypeName<T> << " | "
              << std::right << std::setw(w3) << 8 * sizeof(T) << " | "
              << std::right << std::setw(w4) << sizeof(T) << " \n";
        };
        
        s << "\n";
        s << std::right << std::setw(w0) << "Internal name" << " | "
          << std::right << std::setw(w1) << "TypeName" << " | "
          << std::right << std::setw(w2) << "FullTypeName" << " | "
          << std::right << std::setw(w3) << "Bits" << " | "
          << std::right << std::setw(w4) << "Bytes" << " \n";
       
        s << std::string( w0, '-' ) << "-+-"
          << std::string( w1, '-' ) << "-+-"
          << std::string( w2, '-' ) << "-+-"
          << std::string( w3, '-' ) << "-+-"
          << std::string( w4, '-' ) << "-\n";
        
        line.operator()<                  Int8>( "                  Int8" );
        line.operator()<                 Int16>( "                 Int16" );
        line.operator()<                 Int32>( "                 Int32" );
        line.operator()<                 Int64>( "                 Int64" );
        
        line.operator()<                 UInt8>( "                 UInt8" );
        line.operator()<                UInt16>( "                UInt16" );
        line.operator()<                UInt32>( "                UInt32" );
        line.operator()<                UInt64>( "                UInt64" );
        
        
        s << std::string( w0, ' ' ) << " | "
          << std::string( w1, ' ' ) << " | "
          << std::string( w2, ' ' ) << " | "
          << std::string( w3, ' ' ) << " | "
          << std::string( w4, ' ' ) << " \n";
        
        line.operator()<                  bool>( "                  bool" );
        line.operator()<                  char>( "                  char" );
        line.operator()<                size_t>( "                size_t" );
        
        s << std::string( w0, ' ' ) << " | "
          << std::string( w1, ' ' ) << " | "
          << std::string( w2, ' ' ) << " | "
          << std::string( w3, ' ' ) << " | "
          << std::string( w4, ' ' ) << " \n";
        
        line.operator()<  signed          char>( "  signed          char" );
        line.operator()<  signed short     int>( "  signed short     int" );
        line.operator()<  signed           int>( "  signed           int" );
        line.operator()<  signed long      int>( "  signed long      int" );
        line.operator()<  signed long long int>( "  signed long long int" );
        
        s << std::string( w0, ' ' ) << " | "
          << std::string( w1, ' ' ) << " | "
          << std::string( w2, ' ' ) << " | "
          << std::string( w3, ' ' ) << " | "
          << std::string( w4, ' ' ) << " \n";
        
        line.operator()<unsigned          char>( "unsigned          char" );
        line.operator()<unsigned short     int>( "unsigned short     int" );
        line.operator()<unsigned           int>( "unsigned           int" );
        line.operator()<unsigned long      int>( "unsigned long      int" );
        line.operator()<unsigned long long int>( "unsigned long long int" );
        
        s << std::string( w0, ' ' ) << " | "
          << std::string( w1, ' ' ) << " | "
          << std::string( w2, ' ' ) << " | "
          << std::string( w3, ' ' ) << " | "
          << std::string( w4, ' ' ) << " \n";
        
        return s.str();
    }
    
    [[maybe_unused]] std::string FloatingPointInfo()
    {
        std::stringstream s ;

        const int  w0 = 22 + 1;
        const int  w1 = 11;
        const int  w2 = 22;
        const int  w3 = 5;
        const int  w4 = 5;
        const int  w5 = 4;
        const int  w6 = 7;
        
        
        auto line = [=,&s]<typename T>( const char * name )
        {
            s << std::right << std::setw(w0) << name << " | "
              << std::right << std::setw(w1) << TypeName<T> << " | "
              << std::right << std::setw(w2) << FullTypeName<T> << " | "
              << std::right << std::setw(w3) << 8 * sizeof(T) << " | "
              << std::right << std::setw(w4) << sizeof(T) << " | "
              << std::right << std::setw(w5) << TypeName<Scalar::Real<T>> << " | "
              << std::right << std::setw(w6) << TypeName<Scalar::Complex<T>> << " \n";
        };
        
        s << "\n";
        s << std::right << std::setw(w0) << "Internal name" << " | "
          << std::right << std::setw(w1) << "TypeName" << " | "
          << std::right << std::setw(w2) << "FullTypeName" << " | "
          << std::right << std::setw(w3) << "Bits" << " | "
          << std::right << std::setw(w4) << "Bytes" << " | "
          << std::right << std::setw(w5) << "Real" << " | "
          << std::right << std::setw(w6) << "Complex" << " \n";
       
        s << std::string( w0, '-' ) << "-+-"
          << std::string( w1, '-' ) << "-+-"
          << std::string( w2, '-' ) << "-+-"
          << std::string( w3, '-' ) << "-+-"
          << std::string( w4, '-' ) << "-+-"
          << std::string( w5, '-' ) << "-+-"
          << std::string( w6, '-' ) << "-\n";
        
        line.operator()<                Real32>( "                Real32" );
        line.operator()<                Real64>( "                Real64" );
        line.operator()<             Complex32>( "             Complex32" );
        line.operator()<             Complex64>( "             Complex64" );
        
        s << std::string( w0, ' ' ) << " | "
          << std::string( w1, ' ' ) << " | "
          << std::string( w2, ' ' ) << " | "
          << std::string( w3, ' ' ) << " | "
          << std::string( w4, ' ' ) << " | "
          << std::string( w5, ' ' ) << " | "
          << std::string( w6, ' ' ) << " \n";
        
        return s.str();
    }
    
} // namespace Tools
