#pragma once

namespace Tools
{
    template<typename F, IntQ Int>
    Int FindFirst( F && f, const Int a, const Int b, const Int inc = Int(1) )
    {
        for( Int i = a; i < b; i += inc )
        {
            if( std::invoke(f,i) ) { return i; }
        }
        return b;
    }
    
    template<typename F, IntQ Int>
    Int FindLast( F && f, const Int a, const Int b )
    {
        for( Int i = b; i --> a;  )
        {
            if( std::invoke(f,i) ) { return i; }
        }
        return b;
    }
    
    template<typename F, IntQ Int>
    bool ForAllQ( F && f, const Int a, const Int b, const Int inc = Int(1) )
    {
        for( Int i = a; i < b; i += inc )
        {
            if( !std::invoke(f,i) ) { return false; }
        }
        return true;
    }
    
    template<typename F, IntQ Int>
    bool ExistsQ( F && f, const Int a, const Int b, const Int inc = Int(1) )
    {
        for( Int i = a; i < b; i += inc )
        {
            if( std::invoke(f,i) ) { return true; }
        }
        return false;
    }
}
