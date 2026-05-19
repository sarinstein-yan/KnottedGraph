#pragma once

// Playing around with designated initializers

namespace Tools
{
    struct Kron
    {
        using F_T = Scalar::Flag;
         
        struct TArg
        {
            Scalar::Flag a_flag = F_T::Generic;
            Op opA = Op::Id; Size_T mA = VarSize; Size_T nA = VarSize;
            Op opB = Op::Id; Size_T mB = VarSize; Size_T nB = VarSize;
            Scalar::Flag b_flag = F_T::Generic;
            Parallel_T parQ = Sequential;
        }; // TArg
        
        template<TArg targ>
        struct Prod
        {
            
            template <
                typename A_T, typename B_T, typename C_T,
                typename a_T = Scalar::Real<A_T>,
                typename b_T = Scalar::Real<C_T>,
                IntQ     Int = Size_T
            >
            struct Arg
            {
                // We initialize this to 1 if we have no reason to do otherwise.
                a_T a = (targ.a_flag == F_T::Minus ? a_T(-1) :
                            (targ.a_flag == F_T::Zero ? a_T(0) : a_T(1) ) );
                cptr<A_T> A;
                Int mA = targ.mA;
                Int nA = targ.nA;
                Int ldA = nA;
                cptr<B_T> B;
                Int mB = targ.mB;
                Int nB = targ.nB;
                Int ldB = nB;
                // We initialize this to 0 if we have no reason to do otherwise.
                b_T b  = (targ.b_flag == F_T::Minus ? b_T(-1) :
                             (targ.b_flag == F_T::Plus ? b_T(1) : b_T(0) ) );
                mptr<C_T> C;
                Int ldC = nA * nB;
                Int thread_count = 1;
            }; // Arg
            
            
//            // deduction guide
//            template <
//                typename A_T, typename B_T, typename C_T,
//                typename a_T, typename b_T, IntQ Int
//            >
//            Arg( a_T, A_T const *, Int, Int, Int, B_T const *, Int, Int, Int, b_T, C_T *, Int, Int )
//            ->
//            Arg<A_T,B_T,C_T,a_T,b_T,Int>;
//
//            template <
//                typename A_T, typename B_T, typename C_T,
//                typename a_T, typename b_T
//            >
//            Arg( a_T, A_T const *, B_T const *, b_T, C_T * )
//            ->
//            Arg<A_T,B_T,C_T,a_T,b_T,Size_T>;
            
            
            template<
                typename A_T, typename B_T, typename C_T,
                typename a_T, typename b_T, IntQ Int
            >
            void operator()( Arg<A_T,B_T,C_T,a_T,b_T,Int> arg )
            {
                constexpr Scalar::Flag A_flag =
                    (targ.a_flag == Scalar::Flag::Zero)
                    ? Scalar::Flag::Zero
                    : Scalar::Flag::Generic;
                
                
                // TODO: Currently we parallelize only over the first dimension of `A`. We have bad luck if that is small.
                
                Do<targ.nA,targ.parQ>(
                    [=](const Int i )
                    {
                        const Int incC = arg.ldC * arg.mB * i;
                        
                        for( Int k = 0; k < arg.mB; ++k )
                        {
                            outerprod_buffers<
                                A_flag, targ.b_flag, targ.nA, targ.nB,
                                Sequential, targ.opA, targ.opB
                            >(
                              arg.a, &arg.A[arg.ldA * i],
                                     &arg.B[arg.ldB * k],
                              arg.b, &arg.C[incC + arg.ldC * k],
                              arg.nB, arg.nA, arg.nB
                            );
                        }
                    },
                    arg.mA, arg.thread_count
                                      
                ); // Do
                
            } // operator()
            
        }; // Prod
        
    }; // Kron
    
    struct KronProd_TArgs
    {
        Scalar::Flag a_flag = Scalar::Flag::Generic;
        Op opA = Op::Id; Size_T mA = VarSize; Size_T nA = VarSize;
        Op opB = Op::Id; Size_T mB = VarSize; Size_T nB = VarSize;
        Scalar::Flag b_flag = Scalar::Flag::Generic;
        Parallel_T parQ = Sequential;
    };
    
    template <
        KronProd_TArgs targs,
        typename a_T, typename A_T, typename B_T,
        typename b_T, typename C_T, IntQ Int
    >
    struct KronProd_Args
    {
        cref<a_T> a;
        cptr<A_T> A;
        Int mA = static_cast<Int>(targs.mA);
        Int nA = static_cast<Int>(targs.nA);
        Int ldA = nA;
        cptr<B_T> B;
        Int mB = static_cast<Int>(targs.mB);
        Int nB = static_cast<Int>(targs.nB);
        Int ldB = nB;
        cref<b_T> b;
        mptr<C_T> C;
        Int ldC = nA * nB;
        Int thread_count = 1;
    };
    
    /*! Computes C = a * op_x(x) * op_y(y)^T + b * A, where op_x and op_y may either be `Op::Id` or `Op::Conj`.
     *
     */
    
    template <
        KronProd_TArgs targs,
        typename a_T, typename A_T, typename B_T,
        typename b_T, typename C_T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void KronProd(
        KronProd_Args<targs,a_T,A_T,B_T,b_T,C_T,Int> args
    )
    {
        constexpr Scalar::Flag A_flag =
            (targs.a_flag == Scalar::Flag::Zero)
            ? Scalar::Flag::Zero
            : Scalar::Flag::Generic;
        
        
        // TODO: Currently we parallelize only over the first dimension of `A`. We have bad luck if that is small.
        
        Do<targs.nA,targs.parQ>(
            [=](const Int i )
            {
                const Int incC = args.ldC * args.mB * i;
                
                for( Int k = 0; k < args.mB; ++k )
                {
                    outerprod_buffers<
                        A_flag, targs.b_flag, targs.nA, targs.nB,
                        Sequential, targs.opA, targs.opB
                    >(
                      args.a, &args.A[args.ldA * i],
                              &args.B[args.ldB * k],
                      args.b, &args.C[incC + args.ldC * k],
                      args.nB, args.nA, args.nB
                    );
                }
            },
            args.mA, args.thread_count
        );
    }
    
    
//    // Under clang-19 (not Apple clang!) we can now call Kron, e.g. this way:
//
//    using KronProd = Kron::Prod<{ .parQ = Parallel }>;
//
//    kron(KronProd::Arg{
//        .a = a, .A = A.data(), .mA = A.Dim(0), .nA = A.Dim(1), .ldA = A.Dim(1),
//                .B = B.data(), .mB = B.Dim(0), .nB = B.Dim(1), .ldB = B.Dim(1),
//        .b = b, .C = C.data(),                                 .ldC = C.Dim(1),
//        .thread_count = thread_count
//    });
    
//    // This is still not exactly what I want:
//
//    Kron::Prod<.parQ = Parallel>(
//        .a = a, .A = A.data(), .mA = A.Dim(0), .nA = A.Dim(1), .ldA = A.Dim(1),
//                .B = B.data(), .mB = B.Dim(0), .nB = B.Dim(1), .ldB = B.Dim(1),
//        .b = b, .C = C.data(),                                 .ldC = C.Dim(1),
//        .thread_count = thread_count
//    );
//
//    // But it is as close as I can get to it with C++20.
//
//    // Under Apple clang we still have to give a lot of help:
//
//    using KronProd = Kron::Prod<Kron::TArg{ .parQ = Parallel }>;
//
//    KronProd kron;
//
//    kron(KronProd::Arg{
//        .a = a, .A = A.data(), .mA = A.Dim(0), .nA = A.Dim(1), .ldA = A.Dim(1),
//                .B = B.data(), .mB = B.Dim(0), .nB = B.Dim(1), .ldB = B.Dim(1),
//        .b = b, .C = C.data(),                                 .ldC = C.Dim(1),
//        .thread_count = thread_count
//    });
//
//    // And then it even does not work with deduction guides for each and every case.
        
} // namespace Tools


// A test program that works under clang-19

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
//{
//    using KronProd = Kron::Prod<{ .parQ = Parallel }>;
//
//    KronProd()(KronProd::Arg{
//        .a = a, .A = A.data(), .mA = A.Dim(0), .nA = A.Dim(1), .ldA = A.Dim(1),
//        .B = B.data(), .mB = B.Dim(0), .nB = B.Dim(1), .ldB = B.Dim(1),
//        .b = b, .C = C.data(),                                 .ldC = C.Dim(1),
//        .thread_count = thread_count
//    });
//}
//
//
//TOOLS_DUMP(C);
//
//{
//    using KronProd = Kron::Prod<{ .mA = 3, .nA = 3, .mB = 3, .nB = 3 }>;
//
//    KronProd()(KronProd::Arg{
//        .a = a, .A = A.data(),
//                .B = B.data(),
//        .b = b, .C = C.data()
//    });
//}
//
//
//TOOLS_DUMP(C);
