#pragma once

#include <initializer_list>

namespace Tools
{
    // We cannot return a char array from a function, therefore we need a wrapper
    template <unsigned N>
    struct CompileTimeString
    {
        char c[N];
    };

    template<unsigned ...Len>
    constexpr auto combine_strings( const char (&...strings)[Len] )
    {
        constexpr unsigned N = (... + Len) - sizeof...(Len);
        CompileTimeString<N + 1> result = {};
        result.c[N] = '\0';
        
        char * dst = result.c;
        for (const char* src : {strings...})
        {
            for( ; *src != '\0'; src++, dst++ )
            {
                *dst = *src;
            }
        }
        return result;
    }

}
