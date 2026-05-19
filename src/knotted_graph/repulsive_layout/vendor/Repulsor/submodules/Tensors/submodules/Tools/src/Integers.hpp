namespace Tools
{
    
    using Int8    = std::int8_t;
    using Int16   = std::int16_t;
    using Int32   = std::int32_t;
    using Int64   = std::int64_t;
    
    using UInt8   = std::uint8_t;
    using UInt16  = std::uint16_t;
    using UInt32  = std::uint32_t;
    using UInt64  = std::uint64_t;
    
    using FastInt8  = std::int_fast8_t;
    using FastInt16 = std::int_fast16_t;
    using FastInt32 = std::int_fast32_t;
    using FastInt64 = std::int_fast64_t;
    
    using FastUInt8  = std::uint_fast8_t;
    using FastUInt16 = std::uint_fast16_t;
    using FastUInt32 = std::uint_fast32_t;
    using FastUInt64 = std::uint_fast64_t;
    
    namespace Scalar
    {
        template<typename T> constexpr bool ComplexQ = false;
        template<typename T> constexpr bool RealQ    = false;
        
        
        template<typename T, typename Enable = void>
        struct IntegralTypeTraits
        {
            static constexpr bool         realQ     = false;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "UnknownType";
            static constexpr const char * full_name = "UnknownType";
        };

        template<> struct IntegralTypeTraits<bool>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "Bool";
            static constexpr const char * full_name = "bool";
        };

        template<> struct IntegralTypeTraits<Int8>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "I8";
            static constexpr const char * full_name = "std::int8_t";
        };
        
        template<> struct IntegralTypeTraits<Int16>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "I16";
            static constexpr const char * full_name = "std::int16_t";
        };
        
        template<> struct IntegralTypeTraits<Int32>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "I32";
            static constexpr const char * full_name = "std::int32_t";
        };
        
        template<> struct IntegralTypeTraits<Int64>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "I64";
            static constexpr const char * full_name = "std::int64_t";
        };
        
        template<> struct IntegralTypeTraits<UInt8>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "U8";
            static constexpr const char * full_name = "std::uint8_t";
        };
        
        template<> struct IntegralTypeTraits<UInt16>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "U16";
            static constexpr const char * full_name = "std::uint16_t";
        };
        
        template<> struct IntegralTypeTraits<UInt32>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "U32";
            static constexpr const char * full_name = "std::uint32_t";
        };
        
        template<> struct IntegralTypeTraits<UInt64>
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "U64";
            static constexpr const char * full_name = "std::uint64_t";
        };
        
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,Size_T>
                &&
                !std::is_same_v<T,Int64>
                &&
                !std::is_same_v<T,Int32>
                &&
                !std::is_same_v<T,Int16>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "Size_T";
            static constexpr const char * full_name = "std::size_t";
        };

        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,char>
                &&
                !std::is_same_v<T,Int8>
                &&
                !std::is_same_v<T,UInt8>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "char";
            static constexpr const char * full_name = "char";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,signed char>
                &&
                !std::is_same_v<T,Int8>
                &&
                !std::is_same_v<T,char>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "signed char";
            static constexpr const char * full_name = "signed char";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,unsigned char>
                &&
                !std::is_same_v<T,UInt8>
                &&
                !std::is_same_v<T,char>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "unsigned char";
            static constexpr const char * full_name = "unsigned char";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,int>                
                &&
                !std::is_same_v<T,Int64> // Should actually be forbidden.
                &&
                !std::is_same_v<T,Int32>
                &&
                !std::is_same_v<T,Int16>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "int";
            static constexpr const char * full_name = "int";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,unsigned int>
                &&
                !std::is_same_v<T,UInt64>
                &&
                !std::is_same_v<T,UInt32>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "unsigned int";
            static constexpr const char * full_name = "unsigned int";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,short int>
                &&
                !std::is_same_v<T,Int64> // Should actually be forbidden.
                &&
                !std::is_same_v<T,Int32> // Should actually be forbidden.
                &&
                !std::is_same_v<T,Int16>
                &&
                !std::is_same_v<T,int>
            >
        >
        {
            static constexpr bool       realQ       = true;
            static constexpr bool       complexQ    = false;
            static constexpr const char * type_name = "short int";
            static constexpr const char * full_name = "short int";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,unsigned short int>
                &&
                !std::is_same_v<T,UInt64> // Should actually be forbidden.
                &&
                !std::is_same_v<T,UInt32> // Should actually be forbidden.
                &&
                !std::is_same_v<T,UInt16>
                &&
                !std::is_same_v<T,unsigned  int>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "unsigned short int";
            static constexpr const char * full_name = "unsigned short int";
        };
        
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,long int>
                &&
                !std::is_same_v<T,Int64>
                &&
                !std::is_same_v<T,Int32>
                &&
                !std::is_same_v<T,Size_T>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "L";
            static constexpr const char * full_name = "signed long int";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,long long int>
                &&
                !std::is_same_v<T,Int64>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "LL";
            static constexpr const char * full_name = "signed long long int";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,unsigned long int>
                &&
                !std::is_same_v<T,UInt64>
                &&
                !std::is_same_v<T,UInt32>
                &&
                !std::is_same_v<T,Size_T>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "UL";
            static constexpr const char * full_name = "unsigned long int";
        };
        
        template<typename T>
        struct IntegralTypeTraits<
            T,
            typename std::enable_if_t<
                std::is_same_v<T,unsigned long long int>
                &&
                !std::is_same_v<T,UInt64>
            >
        >
        {
            static constexpr bool         realQ     = true;
            static constexpr bool         complexQ  = false;
            static constexpr const char * type_name = "ULL";
            static constexpr const char * full_name = "unsigned long long int";
        };
        
        template<typename T> requires std::is_integral_v<T>
        static constexpr bool RealQ<T> = IntegralTypeTraits<T>::realQ;
        
        template<typename T> requires std::is_integral_v<T>
        static constexpr bool ComplexQ<T> = IntegralTypeTraits<T>::complexQ;
        
        
    } // namespace Scalar
    
}

