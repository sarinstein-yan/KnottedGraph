public:

template<
    typename Fmt = Format::Cube::Medium,
    IntQ Int_0, IntQ Int_1, IntQ Int_2, ArrayFun<Int_0,Int_1,Int_2> A, CharConv<Result_T<A>> C
>
OutString & PutCubeFun( A && a, C && to_chars, Int_0 d_0, Int_1 d_1, Int_2 d_2, bool check_sizeQ = true )
{
    return PutArray(
        std::forward<A>(a), std::forward<C>(to_chars), check_sizeQ,
        d_0, Fmt::prefix_0, Fmt::infix_0, Fmt::suffix_0,
        d_1, Fmt::prefix_1, Fmt::infix_1, Fmt::suffix_1,
        d_2, Fmt::prefix_2, Fmt::infix_2, Fmt::suffix_2
    );
}

template<
    typename Fmt = Format::Cube::Medium,
    IntQ Int_0, IntQ Int_1, IntQ Int_2, ArrayFun<Int_0,Int_1,Int_2> A
>
OutString & PutCubeFun( A && a, Int_0 d_0, Int_1 d_1, Int_2 d_2, bool check_sizeQ = true )
{
    return PutCubeFun<Fmt>(std::forward<A>(a), ToChars<Result_T<A>>(), d_0, d_1, d_2, check_sizeQ);
}

template<
    typename Fmt = Format::Cube::Medium,
    typename T, IntQ Int_0, IntQ Int_1, IntQ Int_2, CharConv<T> C
>
OutString & PutCube( cptr<T> a, C && to_chars, Int_0 d_0, Int_1 d_1, Int_2 d_2, bool check_sizeQ = true )
{
    return PutCubeFun<Fmt>(
        [a,d_1,d_2]( const Int_0 i_0, const Int_1 i_1, const Int_2 i_2 )
        {
            return a[(d_1 * i_0 + i_1) * d_2 + i_2];
        },
        std::forward<C>(to_chars), d_0, d_1, d_2, check_sizeQ
    );
}

template<
    typename Fmt = Format::Cube::Medium,
    typename T, IntQ Int_0, IntQ Int_1, IntQ Int_2
>
OutString & PutCube( cptr<T> a, Int_0 d_0, Int_1 d_1, Int_2 d_2, bool check_sizeQ = true )
{
    return PutCube<Fmt>(a, ToChars<T>(), d_0, d_1, d_2, check_sizeQ);
}



template<
    typename Fmt = Format::Cube::Medium,
    IntQ Int_0, IntQ Int_1, IntQ Int_2, ArrayFun<Int_0,Int_1,Int_2> A, CharConv<Result_T<A>> C
>
static OutString FromCube( A && a, C && to_chars, Int_0 d_0, Int_1 d_1, Int_2 d_2 )
{
    OutString out { ArrayCharCount(
        std::forward<C>(to_chars),
        d_0, Fmt::prefix_0, Fmt::infix_0, Fmt::suffix_0,
        d_1, Fmt::prefix_1, Fmt::infix_1, Fmt::suffix_1,
        d_2, Fmt::prefix_2, Fmt::infix_2, Fmt::suffix_2
    ) };

    out.PutCubeFun<Fmt>(std::forward<A>(a), std::forward<C>(to_chars), d_0, d_1, d_2, false);
    
    return out;
}

template<
    typename Fmt = Format::Cube::Medium,
    IntQ Int_0, IntQ Int_1, IntQ Int_2, ArrayFun<Int_0,Int_1,Int_2> A
>
static OutString FromCube( A && a, Int_0 d_0, Int_1 d_1, Int_2 d_2 )
{
    return FromCube<Fmt>(std::forward<A>(a), ToChars<Result_T<A>>(), d_0, d_1, d_2);
}

template<
    typename Fmt = Format::Cube::Medium,
    IntQ Int_0, IntQ Int_1, IntQ Int_2, typename T, CharConv<T> C
>
static OutString FromCube( cptr<T> a, C && to_chars, Int_0 d_0, Int_1 d_1, Int_2 d_2 )
{
    return FromCube<Fmt>(
        [a,d_1,d_2]( const Int_0 i_0, const Int_1 i_1, const Int_2 i_2 )
        {
            return a[(d_1 * i_0 + i_1) * d_2 + i_2];
        },
        std::forward<C>(to_chars), d_0, d_1, d_2
    );
}

template<
    typename Fmt = Format::Cube::Medium,
    IntQ Int_0, IntQ Int_1, IntQ Int_2, typename T
>
static OutString FromCube( cptr<T> a, Int_0 d_0, Int_1 d_1, Int_2 d_2 )
{
    return FromCube<Fmt>(a, ToChars<T>(), d_0, d_1, d_2);
}
