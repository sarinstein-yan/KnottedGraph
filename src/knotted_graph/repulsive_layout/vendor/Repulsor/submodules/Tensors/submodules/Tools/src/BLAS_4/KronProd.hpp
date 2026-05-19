#pragma once

// Playing around with designated initializers

namespace Tools
{
     
    struct KronProd_TArg
    {
        Scalar::Flag a_flag = Scalar::Flag::Generic;
        Op opA = Op::Id; Size_T mA = VarSize; Size_T nA = VarSize;
        Op opB = Op::Id; Size_T mB = VarSize; Size_T nB = VarSize;
        Scalar::Flag b_flag = Scalar::Flag::Generic;
        Parallel_T parQ = Sequential;
    };
    
    template<KronProd_TArg t>
    struct KronProd
    {
        template <
            typename A_T, typename B_T, typename C_T,
            typename a_T = Scalar::Real<A_T>,
            typename b_T = Scalar::Real<C_T>,
            IntQ Int = Size_T
        >
        struct Kernel
        {
            using F_T = Scalar::Flag;
         
            // We initialize this to 1 if we have no reason to do otherwise.
            const a_T a   = (t.a_flag == F_T::Minus ? a_T(-1) :
                              (t.a_flag == F_T::Zero ? a_T(0) : a_T(1) ) );
            cptr<A_T> A;
            const Int mA  = t.mA;
            const Int nA  = t.nA;
            const Int ldA = nA;
            cptr<B_T> B;
            const Int mB  = t.mB;
            const Int nB  = t.nB;
            const Int ldB = nB;
            // We initialize this to 0 if we have no reason to do otherwise.
            const b_T b   = (t.b_flag == F_T::Minus ? b_T(-1) :
                              (t.b_flag == F_T::Plus ? b_T(1) : b_T(0) ) );
            mptr<C_T> C;
            const Int ldC = nA * nB;
            const Int thread_count = 1;
            
            void operator()()
            {
                constexpr Scalar::Flag A_flag =
                    (t.a_flag == Scalar::Flag::Zero)
                    ? Scalar::Flag::Zero
                    : Scalar::Flag::Generic;
                
                
                // TODO: Currently we parallelize only over the first dimension of `A`. We have bad luck if that is small.
                
                Do<t.nA,t.parQ>(
                    [=,this](const Int i )
                    {
                        for( Int k = 0; k < mB; ++k )
                        {
                            OuterProd<{
                                .a_flag = A_flag,  .opx = t.opA, .nx = t.nA,
                                                   .opy = t.opB, .ny = t.nB,
                                .b_flag = t.b_flag
                            }>::Kernel{
                                .a  = this->a,
                                .x  = &this->A[this->ldA * i],
                                .nx = this->nA,
                                .y  = &this->B[this->ldB * k],
                                .ny = this->nB,
                                .b  = this->b,
                                .C  = &this->C[this->ldC * (this->mB * i + k)]
                            }();
                        }
                    },
                    mA, thread_count
                                      
                ); // Do
                
            } // operator()
            
        }; // Kernel
        
    }; // KronProd
        
} // namespace Tools

//// A test program that should works under clang-19, but does not... =/
//
//const Int n = 3;
//const Int thread_count = 1;
//
//Tensor2<Real32,Int> A ( n, n );
//Tensor2<Real32,Int> B ( n, n );
//Tensor2<Real32,Int> C ( n * n, n * n );
//
//Real32 a = 2.;
//Real32 b = 0.;
//A.Random();
//B.Random();
//C.Random();
//
//
//OuterProd<{}>::Kernel{
//    .a  = a, .x  = A.data(), .nx = A.Dim(1),
//             .y  = B.data(); .ny = B.Dim(1),
//    .b  = b, .C  = &C[C.Dim(1) * (B.Dim(0) * i + k)]
//}();
//
//
//KronProd<{ .parQ = Parallel }>::Kernel{
//    .a = a, .A = A.data(), .mA = A.Dim(0), .nA = A.Dim(1), .ldA = A.Dim(1),
//            .B = B.data(), .mB = B.Dim(0), .nB = B.Dim(1), .ldB = B.Dim(1),
//    .b = b, .C = C.data(),                                 .ldC = C.Dim(1),
//    .thread_count = thread_count
//}();
//
//TOOLS_DUMP(C);
//
//KronProd<{ .mA = 3, .nA = 3, .mB = 3, .nB = 3 }>::Kernel{
//    .A = A.data(), .B = B.data(), .C = C.data()
//}();
//
//TOOLS_DUMP(C);
