#pragma once

namespace Tools
{
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE Scalar::Real<x_T> norm_1(
        cptr<x_T> x, const Int n = static_cast<Int>(N), const Int thread_count = 1
    )
    {
        using T = Scalar::Real<x_T>;
        
        {
            TOOLS_MAKE_FP_FAST()
            
            return DoReduce<N,parQ>(
                [x]( const Int i ) { return Abs(x[i]); },
                []( const T & value, T & result ) { result += value; },
                Scalar::Zero<T>,
                n, thread_count
            );
        }
    }
    
} // namespace Tools
