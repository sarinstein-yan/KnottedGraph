#pragma once

namespace Tools
{
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename F, IntQ Int = Size_T
    >
    auto MinMaxBy(
        F && f,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    ) -> std::pair<decltype(f(Int(0))),decltype(f(Int(0)))>
    {
        using S = decltype(f(Int(0)));
        
        using T = std::pair<S,S>;
        
        T init { std::numeric_limits<S>::max(), std::numeric_limits<S>::lowest() };
        
//        return DoReduce<N,parQ>(
//            [&f]( const Int i ) -> T
//            {
//                const S f_i = f(i);
//                
//                return T(f_i,f_i);
//            },
//            []( cref<T> value, mref<T> result )
//            {
//                result.first  = Min( value.first,  result.first  );
//                result.second = Max( value.second, result.second );
//            },
//            init, n, thread_count
//        );
        
        return DoReduce<N,parQ>(
            std::forward<F>(f),
            []( cref<S> value, mref<T> result )
            {
                result.first  = Min( value,  result.first  );
                result.second = Max( value, result.second );
            },
            init, n, thread_count
        );
    }
    
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE std::pair<S,S>
    minmax_buffer(
        cptr<S> z,
        const Int n = static_cast<int>(N),
        const Int thread_count = Int(1)
    )
    {
        return MinMaxBy<N,parQ>( [z]( const Int i ){ return z[i]; }, n, thread_count );
    }
    
    
    
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename F, IntQ Int = Size_T
    >
    auto MinBy(
        F && f,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    ) -> decltype(f(Int(0)))
    {
        using T = decltype(f(Int(0)));
        
        return DoReduce<N,parQ>(
            std::forward<F>(f),
            []( cref<T> value, mref<T> result ) { result = Min( result, value ); },
            std::numeric_limits<T>::max(), n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename F, IntQ Int = Size_T
    >
    auto MinimumBy(
        F && f,
        const Int n = static_cast<Int>(N),
        const Int thread_count = 1
    ) -> std::pair<Int,decltype(f(Int(0)))>
    {
        using S = decltype( f(Int(0)) );
        
        using T = std::pair<Int,S>;
        
        T init { Int(0), std::numeric_limits<S>::max() };
        
        return DoReduce<N,parQ>(
            [&f]( const Int i ) -> T
            {
                const S f_i = f(i);
                
                return T{i,f_i};
            },
            []( cref<T> value, mref<T> result )
            {
                if( value.second < result.second ) { result = value; }
            },
            init, n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE S
    min_buffer(
        cptr<S> z,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        return MinBy<N,parQ>(
            [z]( const Int i ){ return z[i]; }, n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE
    Int min_pos_buffer(
        cptr<S> z,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        return MinimumBy<N,parQ>(
            [z]( const Int i ){ return z[i]; }, n, thread_count
        ).first;
    }
    
    
    

    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename F, IntQ Int = Size_T
    >
    auto MaxBy(
        F && f,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    ) -> decltype(f(Int(0)))
    {
        using S = decltype(f(Int(0)));
        
        return DoReduce<N,parQ>(
            std::forward<F>(f),
            []( cref<S> value, mref<S> result )
            {
                result = Max( result, value );
            },
            std::numeric_limits<S>::lowest(), n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename F, IntQ Int = Size_T
    >
    auto MaximumBy(
        F && f,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    ) -> std::pair<Int,decltype(f(Int(0)))>
    {
        using S = decltype(f(Int(0)));
        
        using T = std::pair<Int,S>;
        
        T init { Int(0), std::numeric_limits<S>::lowest() };
        
        return DoReduce<N,parQ>(
            [&f]( const Int i ) -> T { return T{i,f(i)}; },
            []( cref<T> value, mref<T> result )
            {
                if( value.second > result.second ) { result = value; }
            },
            init, n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE
    S max_buffer(
        cptr<S> z,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        return MaxBy<N,parQ>(
            [z]( const Int i ){ return z[i]; }, n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential, typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE
    Int max_pos_buffer(
        cptr<S> z,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        return MaximumBy<N,parQ>(
            [z]( const Int i ){ return z[i]; }, n, thread_count
        ).first;
    }
    
    
} // namespace Tools
