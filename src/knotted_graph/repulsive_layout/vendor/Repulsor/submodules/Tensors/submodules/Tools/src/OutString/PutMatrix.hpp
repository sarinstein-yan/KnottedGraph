public:

//= ToChars<Result_T<A>>
template<
    typename Fmt = Format::Matrix::Tall,
    IntQ Int_0, IntQ Int_1, ArrayFun<Int_0,Int_1> A, CharConv<Result_T<A>> C
>
OutString & PutMatrixFun( A && a, C && to_chars, Int_0 d_0, Int_1 d_1, bool check_sizeQ = true )
{
    return PutArray(
        std::forward<A>(a), std::forward<C>(to_chars), check_sizeQ,
        d_0, Fmt::prefix_0, Fmt::infix_0, Fmt::suffix_0,
        d_1, Fmt::prefix_1, Fmt::infix_1, Fmt::suffix_1
    );
}

template<typename Fmt = Format::Matrix::Tall, IntQ Int_0, IntQ Int_1, ArrayFun<Int_0,Int_1> A>
OutString & PutMatrixFun( A && a, Int_0 d_0, Int_1 d_1, bool check_sizeQ = true  )
{
    return PutMatrixFun<Fmt>(a, ToChars<Result_T<A>>(), d_0, d_1, check_sizeQ );
}

template<typename Fmt = Format::Matrix::Tall, typename T, IntQ Int_0, IntQ Int_1, CharConv<T> C>
OutString & PutMatrix( cptr<T> a, C && to_chars, Int_0 d_0, Int_1 d_1, bool check_sizeQ = true )
{
    return PutMatrixFun<Fmt>(
        [a,d_1]( const Int_0 i_0, const Int_1 i_1 ) -> T { return a[i_0 * d_1 + i_1]; },
        std::forward<C>(to_chars), d_0, d_1, check_sizeQ
    );
}


template<typename Fmt = Format::Matrix::Tall,typename T, IntQ Int_0, IntQ Int_1>
OutString & PutMatrix( cptr<T> a, Int_0 d_0, Int_1 d_1, bool check_sizeQ = true )
{
    return PutMatrix<Fmt>(a, ToChars<T>(), d_0, d_1, check_sizeQ);
}





template<
    typename Fmt = Format::Matrix::Tall,
    IntQ Int_0, IntQ Int_1, ArrayFun<Int_0,Int_1> A, CharConv<Result_T<A>> C
>
static OutString FromMatrix( A && a, C && to_chars, Int_0 d_0, Int_1 d_1 )
{
    OutString out { ArrayCharCount(
        std::forward<C>(to_chars),
        d_0, Fmt::prefix_0, Fmt::infix_0, Fmt::suffix_0,
        d_1, Fmt::prefix_1, Fmt::infix_1, Fmt::suffix_1
    ) };
    
    out.PutMatrixFun<Fmt>(std::forward<A>(a), std::forward<C>(to_chars), d_0, d_1, false);
    
    return out;
}

template<typename Fmt = Format::Matrix::Tall, IntQ Int_0, IntQ Int_1, ArrayFun<Int_0,Int_1> A>
static OutString FromMatrix( A && a, Int_0 d_0, Int_1 d_1 )
{
    return FromMatrix<Fmt> (std::forward<A>(a), ToChars<Result_T<A>>(), d_0, d_1);
}

template<typename Fmt = Format::Matrix::Tall, typename T, IntQ Int_0, IntQ Int_1, CharConv<T> C>
static OutString FromMatrix( cptr<T> a, C && to_chars, Int_0 d_0, Int_1 d_1 )
{
    return FromMatrix<Fmt>(
        [a,d_1]( const Int_0 i_0, const Int_1 i_1 ) -> T { return a[i_0 * d_1 + i_1]; },
        std::forward<C>(to_chars), d_0, d_1
    );
}

template<typename Fmt = Format::Matrix::Tall, typename T, IntQ Int_0, IntQ Int_1>
static OutString FromMatrix( cptr<T> a, Int_0 d_0, Int_1 d_1 )
{
    return FromMatrix<Fmt>( a, ToChars<T>(), d_0, d_1 );
}

