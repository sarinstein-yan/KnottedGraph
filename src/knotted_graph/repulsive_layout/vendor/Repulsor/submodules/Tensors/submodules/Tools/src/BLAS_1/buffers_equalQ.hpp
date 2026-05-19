#pragma once

namespace Tools
{
    template< Size_T N = VarSize, typename T, IntQ Int = Size_T>
    TOOLS_FORCE_INLINE bool buffers_equalQ(
        cptr<T> a, cptr<T> b, const Int n = static_cast<Int>(N)
    )
    {
        return ForAllQ( [a,b]( Int i ){ return a[i] == b[i]; }, Int(0), n );
    }
}
