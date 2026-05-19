public:


template<bool allocatedQ = false, NonPointerQ A, NonPointerQ C, typename ...Args>
OutString & PutArray( A && a, C && to_chars, bool check_sizeQ, Args&&... args)
{
    static_assert(C::implementedQ,"");
    
    if constexpr ( allocatedQ )
    {
    }
    else
    {
        if ( check_sizeQ )
        {
            const Size_T full_size = ArrayCharCount(
                std::forward<C>(to_chars), std::forward<Args>(args)...
            );
            RequireFreeSpace(full_size);
        }
    }
    
    putArray(
        std::forward<A>(a),
        std::forward<C>(to_chars),
        std::forward<Args>(args)...
     );

    return *this;
}

template<NonPointerQ A, typename ...Args>
OutString & PutArray( A && a, bool check_sizeQ, Args&&... args )
{
    using T = typename std::remove_reference<typename function_traits<A>::return_type>::type;
    
    return PutArray(
        std::forward<A>(a),
        ToChars<T>(),
        check_sizeQ,
        std::forward<Args>(args)...
    );
}



template<
    IntQ Int, ArrayFun<Int> A, CharConv<Result_T<A>> C,
    Stringy Prefix_T, typename ...Args
>
static OutString FromArray( A && a, C && to_chars, Int n, Prefix_T && prefix, Args&&... args )
{
    OutString out { ArrayCharCount(
        std::forward<C>(to_chars), n,
        std::forward<Prefix_T>(prefix), std::forward<Args>(args)...
    ) };
    
    out.PutArray(
        std::forward<A>(a),
        std::forward<C>(to_chars), false, n,
        std::forward<Prefix_T>(prefix), std::forward<Args>(args)...
    );
    
    return out;
}

template< IntQ Int, ArrayFun<Int> A, Stringy Prefix_T, typename ...Args>
static OutString FromArray( A && a, Int n, Prefix_T && prefix, Args&&... args )
{
    return FromArray(
        std::forward<A>(a), ToChars<Result_T<A>>(), n,
        std::forward<Prefix_T>(prefix), std::forward<Args>(args)...
    );
}

private:

template<
    NonPointerQ A, NonIntQ C, IntQ Int,
    Stringy Prefix_T, Stringy Infix_T, Stringy Suffix_T,
    typename ...Args
>
constexpr void putArray(
    A && a, C && to_chars,
    Int n, Prefix_T && prefix, Infix_T  && infix, Suffix_T && suffix,
    Args&&... args
)
{
    PutChars<false>(prefix);
    if( n > Int(0) )
    {
        const Int i = 0;
        putArray( std::bind_front(a,i), std::forward<C>(to_chars), std::forward<Args>(args)... );
    }
    for( Int i = 1; i < n; ++i )
    {
        PutChars<false>(infix);
        putArray( std::bind_front(a,i), std::forward<C>(to_chars), std::forward<Args>(args)... );
    }
    PutChars<false>(suffix);
}


template<NonPointerQ A, NonIntQ C, IntQ Int, Stringy Prefix_T, Stringy Infix_T, Stringy Suffix_T>
constexpr void putArray(
    A && a, C && to_chars,
    Int n, Prefix_T && prefix, Infix_T  && infix, Suffix_T && suffix
)
{
    PutChars<false>(prefix);
    if( n > Int(0) )
    {
        const Int i = 0;
        Put<false>(a(i),std::forward<C>(to_chars));
    }
    for( Int i = 1; i < n; ++i )
    {
        PutWithPrefix<false>(infix,a(i),std::forward<C>(to_chars));
    }
    PutChars<false>(suffix);
}

//template<
//    typename A, typename C
//>
//void putArray(
//    A && a, C && to_chars
//)
//{
//    Put<false>(a(),std::forward<C>(to_chars));
//}



public:


//template<typename T, IntQ Int, typename ...Args>
//static constexpr Size_T ArrayCharCount( Int n, Args&&... args )
//{
//    return arrayCharCount( ToChars<T>(), n, std::forward<Args>(args)... );
//}

template<NonIntQ C, typename ...Args>
static constexpr Size_T ArrayCharCount( C && to_chars, Args&&... args )
{
    return arrayCharCount( std::forward<C>(to_chars), std::forward<Args>(args)... );
}

private:

template<
    NonIntQ C, IntQ Int, Stringy Prefix_T, Stringy Infix_T, Stringy Suffix_T, typename ...Args
>
static constexpr Size_T arrayCharCount(
    C && to_chars,
    Int n, Prefix_T && prefix, Infix_T && infix, Suffix_T && suffix,
    Args&&... args
)
{
    Size_T size = arrayCharCount( std::forward<C>(to_chars), std::forward<Args>(args)...) + CharCount(infix);
    
    return ToSize_T(n) * size + CharCount(prefix) + CharCount(suffix);
}

template<NonIntQ C>
static constexpr Size_T arrayCharCount( C && to_chars )
{
    return to_chars.char_count;
}
