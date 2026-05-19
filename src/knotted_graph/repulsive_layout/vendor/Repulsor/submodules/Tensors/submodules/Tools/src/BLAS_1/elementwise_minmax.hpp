#pragma once

namespace Tools
{
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename y_T, typename z_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE void elementwise_min(
        cptr<x_T> x, cptr<y_T> y, mptr<z_T> z,
        const Size_T n = N,
        const Size_T thread_count = 1
    )
    {
        TOOLS_MAKE_FP_STRICT();
        
        using namespace Scalar;
  
        Do<N,parQ>(
            [x,y,z]( const Size_T i )
            {
                z[i] = Tools::Min( static_cast<z_T>(x[i]), static_cast<z_T>(y[i]) );
            },
            n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename z_T
    >
    TOOLS_FORCE_INLINE void elementwise_min_update(
        cptr<x_T> x, mptr<z_T> lo,
        const Size_T n = N,
        const Size_T thread_count = 1
    )
    {
        TOOLS_MAKE_FP_STRICT();
        
        using namespace Scalar;
        
        Do<N,parQ>(
            [x,lo]( const Size_T i )
            {
                lo[i] = Tools::Min(static_cast<z_T>(x[i]),lo[i]);
            },
            n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename y_T, typename z_T
    >
    TOOLS_FORCE_INLINE void elementwise_max(
        cptr<x_T> x, cptr<y_T> y, mptr<z_T> z,
        const Size_T n = N,
        const Size_T thread_count = 1
    )
    {
        TOOLS_MAKE_FP_STRICT();
        
        Do<N,parQ>(
            [x,y,z]( const Size_T i )
            {
                z[i] = Tools::Max( static_cast<z_T>(x[i]), static_cast<z_T>(y[i]) );
            },
            n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename z_T
    >
    TOOLS_FORCE_INLINE void elementwise_max_update(
        cptr<x_T> x, mptr<z_T> z,
        const Size_T n = N,
        const Size_T thread_count = 1
    )
    {
        TOOLS_MAKE_FP_STRICT();
        
        Do<N,parQ>(
            [x,z]( const Size_T i )
            {
                z[i] = Tools::Max( static_cast<z_T>(x[i]),z[i] );
            },
            n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename y_T, typename z_T
    >
    TOOLS_FORCE_INLINE void elementwise_minmax(
        cptr<x_T> x, cptr<y_T> y, mptr<z_T> lo, mptr<z_T> hi,
        const Size_T n = N,
        const Size_T thread_count = 1
    )
    {
        TOOLS_MAKE_FP_STRICT();
        
        Do<N,parQ>(
            [x,y,lo,hi]( const Size_T i )
            {
                std::tie(lo[i],hi[i]) = Tools::MinMax( static_cast<z_T>(x[i]), static_cast<z_T>(y[i]) );
            },
            n, thread_count
        );
    }
    
    template<
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename x_T, typename z_T
    >
    TOOLS_FORCE_INLINE void elementwise_minmax_update(
        cptr<x_T> x, mptr<z_T> lo, mptr<z_T> hi,
        const Size_T n = N,
        const Size_T thread_count = 1
    )
    {
        TOOLS_MAKE_FP_STRICT()
        
        Do<N,parQ>(
            [x,lo,hi]( const Size_T i )
            {
                lo[i] = Tools::Min(static_cast<z_T>(x[i]),lo[i]);
                hi[i] = Tools::Max(static_cast<z_T>(x[i]),hi[i]);
            },
            n, thread_count
        );
    }

} // namespace Tools
