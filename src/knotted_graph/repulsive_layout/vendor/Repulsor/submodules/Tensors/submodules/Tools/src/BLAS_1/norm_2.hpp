#pragma once

namespace Tools
{

    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE Scalar::Real<x_T> norm_2_squared(
        cptr<x_T> x, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        using T = Scalar::Real<x_T>;
        
        {
            TOOLS_MAKE_FP_FAST()
            
            return DoReduce<N,parQ>(
                [x]( const Int i ) { return AbsSquared(x[i]); },
                []( const T & value, T & result ) { result += value; },
                Scalar::Zero<T>,
                n, thread_count
            );
        }
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE Scalar::Real<x_T> norm_2(
        cptr<x_T> x, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        return Sqrt( norm_2_squared<N,parQ>( x, n, thread_count ) );
    }
    
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void normalize(
        mptr<x_T> x, Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        const Scalar::Real<x_T> a = Inv(norm_2<N,parQ>(x,n,thread_count));
        
        scale_buffer<N,parQ>(a,x,n,thread_count);
    }
    
    
    
} // namespace Tools
