public:

template<bool checkQ = true>
OutString & PutChar( const char & c )
{
    if constexpr( checkQ ) { RequireFreeSpace( 1 ); }
    buffer[size] = c;
    ++size;
    return *this;
}


template<bool checkQ = true>
OutString & PutChars()
{
    return *this;
}

template<bool checkQ = true>
OutString & PutChars( const char * c, const Size_T n )
{
    if constexpr( checkQ ) { RequireFreeSpace( n ); }
    std::copy_n(c, n, &buffer[size]);
    size += n;
    return *this;
}

template<bool checkQ = true, Size_T n>
OutString & PutChars( const char(&c)[n] )
{
    return PutChars<checkQ>( &c[0], n-1 );
}

//template<bool checkQ = true>
//OutString & PutChars( const std::string & s )
//{
//    return PutChars<checkQ>( &s[0], s.size() );
//}
//
//template<bool checkQ = true>
//OutString & PutChars( std::string && s )
//{
//    return PutChars<checkQ>( &s[0], s.size() );
//}

template<bool checkQ = true>
OutString & PutChars( std::string_view s )
{
    return PutChars<checkQ>( &s[0], s.size() );
}


template<bool checkQ = true>
OutString & Put( std::string_view s )
{
    return PutChars<checkQ>( &s[0], s.size() );
    
//    if constexpr( checkQ ) { RequireFreeSpace( s.size() ); }
//    std::copy_n( &s[0], s.size(), ptr);
//    ptr += s.size();
//    return *this;
}

template<bool checkQ = true, typename T, typename ToChars_T = ToChars<T>>
OutString & Put( const T & x, ToChars_T && to_chars = ToChars_T()  )
{
    if constexpr( checkQ )
    {
        while( TryEmplace(x,std::forward<ToChars_T>(to_chars)) ) { Expand(); }
    }
    else
    {
        (void)TryEmplace(x,std::forward<ToChars_T>(to_chars));
    }
    return *this;
}

template<bool checkQ = true, typename T, typename ToChars_T = ToChars<T>>
OutString & PutWithPrefix( std::string_view s, const T & x, ToChars_T && to_chars = ToChars_T() )
{
    if constexpr( checkQ ) { RequireFreeSpace( to_chars.char_count + s.size() ); }
    Put<false>(s);
    Put<false>(x,std::forward<ToChars_T>(to_chars));
    return *this;
}

template<bool checkQ = true, typename T, typename ToChars_T = ToChars<T>>
OutString & PutWithSuffix( const T & x, std::string_view s, ToChars_T && to_chars = ToChars_T() )
{
    if constexpr( checkQ ) { RequireFreeSpace( to_chars.char_count + s.size() ); }
    Put<false>(x,std::forward<ToChars_T>(to_chars));
    Put<false>(s);
    return *this;
}



//template<bool checkQ = true, typename T, typename ToChars_T = ToChars<T>>
//OutString & PutWithSuffix( const T & x, char c, ToChars_T && to_chars = ToChars_T() )
//{
//    if constexpr( checkQ ) { RequireFreeSpace( to_chars.CharCount(x) + Size_T(1) ); }
//    (void)TryEmplace(x,std::forward<ToChars_T>(to_chars));  // Will always be true, unless allocation fails.
//    PutChar<false>(c);
//    return *this;
//}

//template<bool checkQ = true, typename T, Size_T n, typename ToChars_T = ToChars<T>>
//OutString & PutWithSuffix( const T & x, const char(&c)[n], ToChars_T && to_chars = ToChars_T() )
//{
//    if constexpr( checkQ ) { RequireFreeSpace( to_chars.char_count + n - Size_T(1) ); }
//    Put     <false>(x,std::forward<ToChars_T>(to_chars));
//    PutChars<false>(c);
//    return *this;
//}


//template<bool checkQ = true, typename T, typename ToChars_T = ToChars<T>>
//OutString & PutWithPrefix( char c, const T & x, ToChars_T && to_chars = ToChars_T() )
//{
//    if constexpr( checkQ ) { RequireFreeSpace( to_chars.char_count + Size_T(1) ); }
//    PutChar<false>(c);
//    Put    <false>(x,std::forward<ToChars_T>(to_chars));
//    return *this;
//}

template<bool checkQ = true, typename T, Size_T n, typename ToChars_T = ToChars<T>>
OutString & PutWithPrefix( const char(&c)[n], const T & x, ToChars_T && to_chars = ToChars_T() )
{
    if constexpr( checkQ ) { RequireFreeSpace( to_chars.char_count + n - Size_T(1) ); }
    PutChars<false>(c);
    Put     <false>(x,std::forward<ToChars_T>(to_chars));
    return *this;
}
