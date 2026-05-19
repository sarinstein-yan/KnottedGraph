#pragma once


#include "src/Dense/MatrixBlockMajor.hpp"
#include "src/Dense/BLAS_3.hpp"

namespace Tensors
{
    namespace Dense
    {
//        static constexpr Op O_Id        = Op::Id;
//        static constexpr Op O_Trans     = Op::Trans;
//        static constexpr Op O_Conj      = Op::Conj;
//        static constexpr Op O_ConjTrans = Op::ConjTrans;
//
//        using Flag = Scalar::Flag;
//
//        static constexpr Flag F_Plus    = Flag::Plus;
//        static constexpr Flag F_Minus   = Flag::Minus;
//        static constexpr Flag F_Gen     = Flag::Generic;
//        static constexpr Flag F_Zero    = Flag::Zero;
        
        template<
            Layout layout, Op opA, Op opB,
            typename alpha_T, typename beta_T,
            typename A_T, typename B_T, typename C_T
        >
        TOOLS_FORCE_INLINE void gemm(
            const Size_T m, const Size_T n, const Size_T k,
            cref<alpha_T> alpha, cptr<A_T> A, const Size_T ldA,
                                 cptr<B_T> B, const Size_T ldB,
            cref<beta_T>  beta,  mptr<C_T> C, const Size_T ldC,
            Size_T thread_count
        )
        {
            // TODO: Layout
            
            Dense::BLAS_3<VarSize,VarSize,VarSize,16,8,4,opA,opB,Op::Id,A_T,B_T,C_T,Size_T> blas3 (
                Size_T(m), Size_T(n), Size_T(k), thread_count
            );
            
            blas3.gemm(
                alpha, A, Size_T(k),
                       B, Size_T(n),
                beta,  C, Size_T(n)
            );
        }
    }
}



