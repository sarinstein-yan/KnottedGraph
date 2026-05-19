
InString & SkipWhiteSpace()
{
    if( failedQ ) return *this;
    
    while( !EmptyQ() && std::isspace(*ptr) ) { ++ptr; }
    
//    if( EmptyQ() ) { failedQ = true; }
    
    return *this;
}

InString & SkipChar( const char & c )
{
    if( failedQ ) return *this;
    
    if( !EmptyQ() && (c == *ptr) )
    {
        ++ptr;
    }
    else
    {
        failedQ = true;
    }
    return *this;
}

InString & SkipChars( const char * c, const Size_T n )
{
    if( failedQ ) return *this;
    
    if( (ptr + n <= end) && (std::strncmp(c, ptr, n) == 0) )
    {
        ptr += n;
    }
    else
    {
        failedQ = true;
    }
    return *this;
}

template<Size_T n>
InString & SkipChars( const char(&c)[n] )
{
    return SkipChars( &c[0], n-1 );
}

template<typename T, typename FromChars_T = FromChars<T>>
InString & Take( T & x, FromChars_T && from_chars = FromChars_T() )
{
    if( failedQ ) return *this;
    
    auto r = from_chars( ptr, end, x );
    
    if( r.failedQ )
    {
        failedQ = true;
    }
    else
    {
        ptr = r.ptr;
    }
    return *this;
}
