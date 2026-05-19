#pragma once

namespace Tools
{
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE Int
    iamax_buffer(
        cptr<S> z,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        // Find the first position i such that z[i] has maximal modulus.
        
        if constexpr ( Scalar::RealQ<S> )
        {
            return MaximumBy<N,parQ>(
                [z]( const Int i ){ return Abs(z[i]); }, n, thread_count
            ).first;
        }
        else
        {
            return MaximumBy<N,parQ>(
                [z]( const Int i ){ return AbsSquared(z[i]); }, n, thread_count
            ).first;
        }
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename S, IntQ Int = Size_T
    >
    [[nodiscard]] TOOLS_FORCE_INLINE Int
    iamin_buffer(
        cptr<S> z,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        // Find the first position i such that z[i] has minimal modulus.
        
        if constexpr ( Scalar::RealQ<S> )
        {
            return MinimumBy<N,parQ>(
                [z]( const Int i ){ return Abs(z[i]); }, n, thread_count
            ).first;
        }
        else
        {
            return MinimumBy<N,parQ>(
                [z]( const Int i ){ return AbsSquared(z[i]); }, n, thread_count
            ).first;
        }
    }
    
} // namespace Tools
