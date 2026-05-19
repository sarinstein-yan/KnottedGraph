#pragma once

namespace Tools
{
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE Scalar::Real<x_T> norm_max(
        cptr<x_T> x, const Int n = static_cast<Int>(N), const Int thread_count = Int(1)
    )
    {
        if constexpr ( Scalar::RealQ<x_T> )
        {
            return MaxBy<N,parQ>( [x]( const Int i ){ return Abs(x[i]); }, n, thread_count );
        }
        else
        {
            return Sqrt(
                MaxBy<N,parQ>( [x]( const Int i ){ return AbsSquared(x[i]); }, n, thread_count )
            );
        }
    }
    
} // namespace Tools


