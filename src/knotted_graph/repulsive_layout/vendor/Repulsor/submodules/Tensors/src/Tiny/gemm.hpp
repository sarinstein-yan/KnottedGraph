#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal_A, typename Scal_B, typename Scal_C, IntQ Int
        >
        void gemm(
            cref<Scal_C> alpha,
            cptr<Scal_A>  A_,    const Int ldA,
            cptr<Scal_B>  B_,    const Int ldB,
            cref<Scal_C> beta,
            mptr<Scal_C>  C_,    const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scal_A,Int> A;
            Tiny::Matrix<K,N,Scal_B,Int> B;
            Tiny::Matrix<M,N,Scal_C,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<Overwrite>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal_A, typename Scal_B, typename Scal_C, IntQ Int
        >
        void gemm(
            cref<Scal_C> alpha,
            cptr<Scal_A> A_,    const Int ldA,
            cptr<Scal_B> B_,    const Int ldB,  cptr<Int> idx,
            cref<Scal_C> beta,
            mptr<Scal_C> C_,    const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Reads from B in row-scattered fashion.
            
            Tiny::Matrix<M,K,Scal_A,Int> A;
            Tiny::Matrix<K,N,Scal_B,Int> B;
            Tiny::Matrix<M,N,Scal_C,Int> C;
            
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB,idx);
            Dot<Overwrite>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal_A, typename Scal_B, typename Scal_C, IntQ Int
        >
        void gemm(
            cref<Scal_A> alpha,
            cptr<Scal_A> A_,    const Int ldA,
            cptr<Scal_B> B_,    const Int ldB,
            cref<Scal_C> beta,
            mptr<Scal_C> C_,    const Int ldC,  cptr<Int> idx
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Writes to C in row-scattered fashion.

            Tiny::Matrix<M,K,Scal_A,Int> A;
            Tiny::Matrix<K,N,Scal_B,Int> B;
            Tiny::Matrix<M,N,Scal_C,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<Overwrite>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC,idx);
        }
        
        
    } // namespace Tiny
    
} // namespace Tensors
