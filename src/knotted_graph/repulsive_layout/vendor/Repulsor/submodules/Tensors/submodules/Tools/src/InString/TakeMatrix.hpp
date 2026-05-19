public:

template<
    typename MatrixFun_T, IntQ Int,
    const Size_T n_prefix_0,
    const Size_T n_infix_0,
    const Size_T n_suffix_0,
    const Size_T n_prefix_1,
    const Size_T n_infix_1,
    const Size_T n_suffix_1,
    typename FromChars_T = FromChars<typename std::remove_reference<typename function_traits<MatrixFun_T>::return_type>::type>
>
InString & TakeMatrixFunction( MatrixFun_T && a, Int m, Int n,
    const char(&prefix_0)[n_prefix_0],
    const char(&infix_0 )[n_infix_0 ],
    const char(&suffix_0)[n_suffix_0],
    const char(&prefix_1)[n_prefix_1],
    const char(&infix_1 )[n_infix_1 ],
    const char(&suffix_1)[n_suffix_1],
    FromChars_T && from_chars = FromChars_T()
)
{
    SkipChars(prefix_0);
    if( m > Int(0) )
    {
        const Int i = 0;
        SkipChars(prefix_1);
        if( n > Int(0) )
        {
            const Int j = 0;
            Take(a(i,j),std::forward<FromChars_T>(from_chars));
            if( failedQ ) { return *this; }
        }
        for( Int j = 1; j < n; ++j )
        {
            SkipChars(infix_1);
            Take(a(i,j),std::forward<FromChars_T>(from_chars));
            if( failedQ ) { return *this; }
        }
        SkipChars(suffix_1);
    }
    for( Int i = 1; i < m; ++i )
    {
        SkipChars(infix_0);
        SkipChars(prefix_1);
        if( n > Int(0) )
        {
            const Int j = 0;
            Take(a(i,j),std::forward<FromChars_T>(from_chars));
            if( failedQ ) { return *this; }
        }
        for( Int j = 1; j < n; ++j )
        {
            SkipChars(infix_1);
            Take(a(i,j),std::forward<FromChars_T>(from_chars));
            if( failedQ ) { return *this; }
        }
        SkipChars(suffix_1);
    }
    SkipChars(suffix_0);
    
    return *this;
}
