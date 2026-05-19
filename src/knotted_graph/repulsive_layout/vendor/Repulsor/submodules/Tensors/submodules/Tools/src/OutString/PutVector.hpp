public:

template<
    typename Fmt = Format::Vector::Default,
    IntQ Int_0, ArrayFun<Int_0> A, CharConv<Result_T<A>> C
>
OutString & PutVectorFun( A && a, C && to_chars, Int_0 d_0, bool check_sizeQ = true )
{
    return PutArray(
        std::forward<A>(a), std::forward<C>(to_chars), check_sizeQ, d_0,
        Fmt::prefix_0, Fmt::infix_0, Fmt::suffix_0
    );
}

template<typename Fmt = Format::Vector::Default, IntQ Int_0, ArrayFun<Int_0> A>
OutString & PutVectorFun( A && a, Int_0 d_0, bool check_sizeQ = true )
{
    return PutVectorFun<Fmt>(std::forward<A>(a), ToChars<Result_T<A>>(), d_0, check_sizeQ );
}

template<typename Fmt = Format::Vector::Default, typename T, IntQ Int_0, CharConv<T> C>
OutString & PutVector( cptr<T> a, C && to_chars, Int_0 d_0, bool check_sizeQ = true )
{
    return PutVectorFun<Fmt>( [a]( const Int_0 i_0 ) { return a[i_0]; }, std::forward<C>(to_chars), d_0, check_sizeQ );
}

template<typename Fmt = Format::Vector::Default, typename T, IntQ Int_0>
OutString & PutVector( cptr<T> a, Int_0 d_0, bool check_sizeQ = true )
{
    return PutVectorFun<Fmt>( [a]( const Int_0 i_0 ) { return a[i_0]; }, ToChars<T>(), d_0, check_sizeQ );
}



template<
    typename Fmt = Format::Vector::Default,
    IntQ Int_0, ArrayFun<Int_0> A, CharConv<Result_T<A>> C
>
static OutString FromVector( A && a, C && to_chars, Int_0 d_0 )
{
    OutString out { ArrayCharCount(
        std::forward<C>(to_chars),
        d_0, Fmt::prefix_0, Fmt::infix_0, Fmt::suffix_0
    )};
    
    out.PutVectorFun<Fmt>( std::forward<A>(a), std::forward<C>(to_chars), d_0, false );
    
    return out;
}

template<typename Fmt = Format::Vector::Default, IntQ Int_0, ArrayFun<Int_0> A>
static OutString FromVector( A && a, Int_0 d_0 )
{
    return FromVector<Fmt>(std::forward<A>(a), ToChars<Result_T<A>>(), d_0);
}

template<typename Fmt = Format::Vector::Default,  typename T, IntQ Int_0, CharConv<T> C>
static OutString FromVector( cptr<T> a, C && to_chars, Int_0 d_0 )
{
    return FromVector<Fmt>([a]( const Int_0 i_0 ) { return a[i_0]; }, std::forward<C>(to_chars), d_0);
}

template<typename Fmt = Format::Vector::Default, typename T, IntQ Int_0>
static OutString FromVector( cptr<T> a, Int_0 d_0 )
{
    return FromVector<Fmt>([a]( const Int_0 i_0 ) { return a[i_0]; }, ToChars<T>(), d_0);
}
