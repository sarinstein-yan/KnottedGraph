#pragma once

namespace Tools
{
    /*! Computes C = a * op_x(x) * op_y(y)^T + b * A, where op_x and op_y may either be `Op::Id` or `Op::Conj`.
     *
     */
    
    template <
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Size_T MA = VarSize, Size_T NA = VarSize,
        Size_T MB = VarSize, Size_T NB = VarSize,
        Parallel_T parQ = Sequential,
        Op op_A = Op::Id, Op op_B = Op::Id,
        typename a_T,  typename A_T, typename B_T,
        typename b_T, typename C_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void kron_prod(
        cref<a_T> a, cptr<A_T> A, const Int ldA,
                     cptr<B_T> B, const Int ldB,
        cref<b_T> b, mptr<C_T> C, const Int ldC,
        const Int mA = static_cast<Int>(MA),
        const Int nA = static_cast<Int>(NA),
        const Int mB = static_cast<Int>(MB),
        const Int nB = static_cast<Int>(NB),
        const Int thread_count = Int(1)
    )
    {
//        const Int mC = mA * mB;
//        const Int nC = nA * nB;
        
        constexpr Scalar::Flag A_flag =
            (a_flag == Scalar::Flag::Zero)
            ? Scalar::Flag::Zero
            : Scalar::Flag::Generic;
        
        
        // TODO: Currently we parallelize only over the first dimension of `A`. We have bad luck if that is small.
        
        Do<MA,parQ>(
            [=](const Int i )
            {
                const Int incC = ldC * mB * i;
                
                for( Int k = 0; k < mB; ++k )
                {
                    outerprod_buffers<
                        A_flag,b_flag,NA,NB,Sequential,op_A,op_B
                    >(
                        a, &A[ldA * i], &B[ldB * k], b, &C[incC + ldC * k], nB, nA, nB
                    );
                }
            },
            mA, thread_count
        );
    }
        
} // namespace Tools
