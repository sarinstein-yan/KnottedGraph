#pragma once

namespace Tools
{    
    struct ToCharResult
    {
        char * ptr = nullptr;
        bool failedQ = false;
    };
    
    template<typename T>
    struct ToChars
    {
        static constexpr bool implementedQ = false;
    };

    template<typename T>
    struct ToChars<T&> : public ToChars<T>{};
        
    template<IntQ T>
    struct ToChars<T>
    {
        static constexpr bool implementedQ = true;
        
        static constexpr Size_T char_count = std::numeric_limits<T>::digits10 + std::is_signed_v<T>;

        ToCharResult operator()( char * begin, char * end, const T & x ) const
        {
            auto r = std::to_chars( begin, end, x, 10 );
            return ToCharResult{ .ptr = r.ptr, .failedQ = (r.ec == std::errc::value_too_large)};
        }
    };
    
    
    template<> struct ToChars<bool>
    {
        static constexpr bool implementedQ = true;
        
        static constexpr Size_T char_count = 1;

        ToCharResult operator()( char * begin, char * end, bool x ) const
        {
            if( begin < end )
            {
                *begin = x ? '1' : '0';
                return ToCharResult{ .ptr = begin + 1, .failedQ = false };
            }
            else
            {
                return ToCharResult{ .ptr = begin, .failedQ = true };
            }
        }
    };
    
    
    template<EnumClass T>
    struct ToChars<T>
    {
        using U = std::underlying_type_t<T>;
        
        static constexpr bool implementedQ = true;
        
        static constexpr Size_T char_count = std::numeric_limits<U>::digits10 + std::is_signed_v<U>;

        ToCharResult operator()( char * begin, char * end, const T & x ) const
        {
            auto r = std::to_chars( begin, end, ToUnderlying(x), 10 );
            return ToCharResult{ .ptr = r.ptr, .failedQ = (r.ec == std::errc::value_too_large)};
        }
    };
    
    template<std::floating_point T>
    struct ToChars<T>
    {
        static constexpr bool implementedQ = true;
        
        static constexpr Size_T char_count = std::numeric_limits<T>::max_digits10 + Size_T(7);
        
        ToCharResult operator()( char * begin, char * end, const T & x ) const
        {
            auto r = std::to_chars( begin, end, x, std::chars_format::general, std::numeric_limits<T>::max_digits10 );
            return ToCharResult{ .ptr = r.ptr, .failedQ = (r.ec == std::errc::value_too_large)};
        }
    };
    
    template<std::floating_point T>
    struct ToChars<std::complex<T>>
    {
        static constexpr bool implementedQ = true;
        
        static constexpr Size_T char_count = Size_T(2) * ToChars<T>::char_count + Size_T(2);
        
        ToCharResult operator()( char * begin, char * end, const T & x ) const
        {
            if( begin + char_count > end )
            {
                return {.ptr = begin, .failedQ = true};
            }
            else
            {
                auto r = std::to_chars( begin, end, Re(x), std::chars_format::general, std::numeric_limits<T>::max_digits10 );
                char * ptr = r.ptr;
                *ptr = '+';
                ++ptr;
                r = std::to_chars( begin, end, Im(x), std::chars_format::general, std::numeric_limits<T>::max_digits10 );
                *ptr = 'I';
                ++ptr;
                return {.ptr = begin, .failedQ = false};
            }
        }
    };

} // namespace Tools
