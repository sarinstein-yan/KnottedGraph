#pragma once

// Not production ready. Just a few thoughts on the design of a thin wrapper class that would maybe make calling BLAS routines a bit easier.

namespace Tensors
{
    namespace Dense
    {
        template<bool B>
        using EnableIfB = typename std::enable_if<B, int>::type;
        
        // TODO: Need Read and Write from and to other MatrixView.
        
        // TODO: Enable to reset pointer?
        
        // TODO: Allow nullptr?
        
//        A.template Block<M,N>(i,j).Read ( B.template Block<M,N>(k,l) );
//        A.template Block<M,N>(i,j).Write( B.template Block<M,N>(k,l) );
//        
//        A.template Block<M,N>(i,j,m,n).Read ( B.template Block<M,N>(k,l,m,n) );
//        A.template Block<M,N>(i,j,m,n).Write( B.template Block<M,N>(k,l,m,n) );
//        
//        A.Block(i,j,m,n).Read ( B.Block(k,l,m,n) );
//        A.Block(i,j,m,n).Write( B.Block(k,l,m,n) );

        class MatrixView_TArgs
        {
            Size_T M  = VarSize;
            Size_T N  = VarSize;
            Size_T LD = VarSize;
            Tools::Layout layout = Tools::Layout::RowMajor;
        };
        
        template<typename Scal, IntQ Int>
        class MatrixView_Args
        {
            T * a  = nullptr;
            Int m  = 0;
            Int n  = 0;
            Int ld = 0;
        };
        
        template<
            typename Scal_, IntQ Int_,
            MatrixView_TArgs targs = MatrixView_TArgs{}
        >
        {
            class MatrixView final
            {
            public:
                
                using Real   = Real_;
                using Int    = Int_;
                
                static constexpr Int M  = targs.M;
                static constexpr Int N  = targs.N;
                static constexpr Int LD = targs.LD;
                static constexpr Tools::Layout layout = targs.layout;
                
                
                static_assert(
                    (LD == VarSize)
                    ||
                    ( RowMajorQ() ? ( (N == 0) || (LD >= N) ) : (M == 0) || (LD >= M) ) )
                ,"");
                
            private:
                
                const Scal * a = nullptr;
                
                const Int m  = M;
                const Int n  = N;
                const Int ld = LD;
                
            public:
                
                // Make nullptr less likely.
                // Maybe a pain in the ass?
                MatrixView() = delete;

                
                MatrixView( cref<MatrixView_Args> args )
                : a  ( args.a  )
                , m  ( M  <= VarSize ? args.m : M  )
                , n  ( N  <= VarSize ? args.n : N  )
                , ld { LD <= VarSize
                        ? (args.ld <= VarSize ? (RowMajorQ() ? n : m) : args.ld )
                        : LD
                     }
                {}
                

                MatrixView( Scal * a_, Int m_, Int n_, Int ld_ )
                : a  ( a_  )
                , m  ( m_  )
                , n  ( n_  )
                , ld ( ld_ )
                {}
                
//                template<EnableIfB<(M == 0) && (N == 0) && (LD == 0)> = 0>
//                MatrixView( Scal * a_, Int m_, Int n_ )
//                : a  ( a_  )
//                , m  ( m_  )
//                , n  ( n_  )
//                , ld ( (layout == Layout::RowMajor) ? n_, m_ )
//                {}
//                
//                
//                template<EnableIfB<(M > 0) && (N == 0) && (LD == 0)> = 0>
//                MatrixView( Scal * a_, Int n_, Int ld_ )
//                : a  ( a_  )
//                , m  ( M   )
//                , n  ( n_  )
//                , ld ( ld_ )
//                {}
//                
//                template<EnableIfB<(M == 0) && (N > 0) && (LD == 0)> = 0>
//                MatrixView( Scal * a_, Int m_, Int ld_ )
//                : a  ( a_  )
//                , m  ( m_  )
//                , n  ( N   )
//                , ld ( ld_ )
//                {}
//                
//                
//                template<EnableIfB<(M > 0) && (N > 0) && (LD == 0)> = 0>
//                MatrixView( Scal * a_, Int ld_ )
//                : a  ( a_  )
//                , m  ( M   )
//                , n  ( N   )
//                , ld ( ld_ )
//                {}
//                
//                
//                template<EnableIfB<(M > 0) && (N > 0) && (LD > 0)> = 0>
//                MatrixView( Scal * a_ )
//                : a  ( a_ )
//                , m  ( M  )
//                , n  ( N  )
//                , ld ( LD )
//                {}
//                
//
//                
//                template<EnableIfB<(M == 0) && (N == 0) && (LD > 0)> = 0>
//                MatrixView( Scal * a_, Int m_, Int n_ )
//                : a  ( a_ )
//                , m  ( m_ )
//                , n  ( n_ )
//                , ld ( LD )
//                {}
//                
//                template<EnableIfB<(M > 0) && (N == 0) && (LD > 0)> = 0>
//                MatrixView( Scal * a_, Int n_ )
//                : a  ( a_ )
//                , m  ( M  )
//                , n  ( n_ )
//                , ld ( LD )
//                {}
//                
//                template<EnableIfB<(M == 0) && (N > 0) && (LD > 0)> = 0>
//                MatrixView( Scal * a_, Int m_ )
//                : a  ( a_ )
//                , m  ( m_ )
//                , ld ( LD )
//                {}
                
                ~MatrixView() = default;
                
                

                Int RowCount() const
                {
                    return (M > Int(0)) ? M : m;
                }
                
                Int ColCount() const
                {
                    return (N > Int(0)) ? N : n;
                }
                
                
                Int LeadingDimension() const
                {
                    return (LD > Int(0)) ? LD : ld;
                }
                
                static constexpr Tools::Layout Layout()
                {
                    return layout;
                }
                
                static constexpr bool RowMajorQ()
                {
                    return (layout == Tools::Layout::RowMajor);
                }
                
                static constexpr bool ColMajorQ()
                {
                    return (layout == Tools::Layout::ColMajor);
                }
                
                
                
                cref<Scal> operator()( const Int i, const Int j ) const
                {
                    if constexpr ( L == Layout::RowMajor )
                    {
                        return a[ld * i + j];
                    }
                    else
                    {
                        return a[ld * j + i];
                    }
                }
                
                mref<Scal> operator()( const Int i, const Int j )
                {
                    if constexpr ( L == Layout::RowMajor )
                    {
                        return a[ld * i + j];
                    }
                    else
                    {
                        return a[ld * j + i];
                    }
                }

                mptr<Scal> data()
                {
                    return &a[0];
                }
                
                cptr<Scal> data() const
                {
                    return &a[0];
                }
                
//                mptr<Scal> data( const Int i )
//                {
//                    
//                    return &a[ld * i];
//                }
//                
//                cptr<Scal> data( const Int i,  ) const
//                {
//                    return &a[ld * i];
//                }
                
                mptr<Scal> data( const Int i, const Int j )
                {
                    if constexpr ( L == Layout::RowMajor )
                    {
                        return &a[ld * i + j];
                    }
                    else
                    {
                        return &a[ld * j + i];
                    }
                }
                
                cptr<Scal> data( const Int i, const Int j ) const
                {
                    if constexpr ( L == Layout::RowMajor )
                    {
                        return &a[ld * i + j];
                    }
                    else
                    {
                        return &a[ld * j + i];
                    }
                }
                
                
//                MatrixView<
//                    Real, Int,
//                    MatrixView_Targs{ .N=N, .M=M, .LD=LD, .layout=Transpose(layout)}
//                >
//                Transpose() const
//                {
//                    return MatrixView<
//                        Real,Int,
//                        MatrixView_Targs{.N=N, .M=M, .LD=LD, .layout=Transpose(layout)}
//                    >( a, n, m, ld );
//                }
                
                template<Size_T RC = VarSize, Size_T CC = VarSize>
                MatrixView<
                    Real,Int,
                    MatrixView_Targs{.N=RC, .M=CC, .LD=LD, .layout=layout}
                >
                Block(
                    const Int i, const Int j,
                    const Int row_count = static_cast<Int>(RC),
                    const Int col_count = static_cast<Int>(CC)
                ) const
                {
                    return MatrixView_Args<Real,Int>
                        {.a=data(i,j), .m=row_count, .n=col_count, .ld=ld};
                    
                }
                
                VectorView<Real,Int,N,RowMajorQ()?1:LD> Row( const Int i ) const
                {
                    if constexpr ( RowMajorQ() )
                    {
                        return VectorView<Real,Int,N,1>(
                            &a[ld * i], col_count, Int(1)
                        );
                    }
                    else
                    {
                        return VectorView<Real,Int,N,LD>(
                            &a[i], col_count, ld
                        );
                    }
                }
                
                VectorView<Real,Int,M,ColMajorQ()?1:LD> Column( const Int j ) const
                {
                    if constexpr ( ColMajorQ() )
                    {
                        return VectorView<Real,Int,M,1>(
                            &a[ld * j], row_count, Int(1)
                        );
                    }
                    else
                    {
                        return VectorView<Real,Int,M,LD>(
                            &a[j], row_count, ld
                        );
                    }
                }
                
                // TODO: Multi-row and multi-col slices?


            } // class MatrixView
        }
        
        
//        // usage example:
//        using Matrix_T = MatrixView<Real,Int>;
//        Matrix_T A ( A_ptr, m_A, n_A, ld_A ); // ld_A can be ignored if contiguous
//        Matrix_T B ( B_ptr, m_B, n_B, ld_B );
//        Matrix_T C ( C_ptr, m_C, n_C, ld_C );
//
//        Dense::gemm<Op::ConjTrans,Op::Id>( alpha, A, B, beta, C );
//
//        vs.
//
//        BLAS::gemm<Layout::RowMajor,Op::ConjTrans,Op::Id>( m_C, n_C, A_m,
//           alpha, A_ptr, ld_A,
//                  B_ptr, ld_B,
//           beta,  C_ptr, ld_C
//        );
//
//
//        Alternative construction syntax:
//        Matrix_T A ({.a=A_ptr, .m=m_A, .n=n_A}); // ld_A can be ignored if contiguous
//
//        Another possible alternative construction syntax:
//        Matrix_T A ({.m=m_A, .n=n_A});
//        A.SetPointer(A_ptr);
//        
//
//        PROS of Dense::gemm:
//            - We can activate many checks.
//            - The dimension k in BLAS::gemm is easy to confuse.
//            - We could reuse an instance of the wrapper to load various blocks of the same size.
//
//        CONS of Dense::gemm:
//            - actually much more verbose
//            - a LOT of wrapper code is necessary to wrap BLAS & LAPACK routines that already have decent call conventions.
//
        
        
        //        A.Block(i,j,m,n).Read ( B.Block(k,l,m,n) );
        //        A.Block(i,j,m,n).Write( B.Block(k,l,m,n) );
        
        
        // Compute C = alpha * op_A(A) * op_B(B) + beta * C;
        
        template<
            Op op_A, Op op_B,
            bool checkQ = false,
            typename T_a, typename T_A, typename I_A, MatrixView_TArgs targs_A,
                          typename T_B, typename I_B, MatrixView_TArgs targs_B,
            typename T_b, typename T_C, typename I_C, MatrixView_TArgs targs_C,
        >
        void gemm(
            T_a alpha, cref<MatrixView<T_A,I_A,targs_A>> A,
                       cref<MatrixView<T_B,I_B,targs_B>> B,
            T_b beta,  mref<MatrixView<T_C,I_C,targs_C>> C
        )
        {
            constexpr Op op_A_ = (A.Layout() == C.Layout()) ? op_A : Transpose(op_A);
            constexpr Op op_B_ = (B.Layout() == C.Layout()) ? op_B : Transpose(op_B);
            
            // TODO: Add compile time dimension checks.
            
            if constexpr ( checkQ )
            {
                // TODO: Add dimension checks.
            }
            
            template<C.Layout(), op_A_, op_B_>
            BLAS::gemm(
                C.RowCount(),
                C.ColCount(),
                NotTransposedQ(op_A_) ? A.ColCount() : A.RowCount(),
                alpha, A.data(), A.LeadingDimension(),
                       B.data(), B.LeadingDimension(),
                beta,  C.data(), C.LeadingDimension()
            );
        }
        
        
        // Compute B = alpha * op_A(A) + beta * B;
        template<
            Scalar::Flag F_a, Scalar::Flag F_b, Op op_A,
            bool checkQ = false,
            typename T_a, typename T_A, typename I_A, MatrixView_TArgs targs_A,
            typename T_b, typename T_B, typename I_B, MatrixView_TArgs targs_B
        >
        void geadd(
            T_a alpha, cref<MatrixView<T_A,I_A,targs_A>> A,
            T_b beta,  mref<MatrixView<T_B,I_B,targs_B>> B
        )
        {
            // TODO: Implement this.
        }
        
        // Compute C = alpha * op_A(A) + beta * B;
        template<
            Scalar::Flag F_a, Scalar::Flag F_b, Op op_A,
            bool checkQ = false,
            typename T_a, typename T_A, typename I_A, MatrixView_TArgs targs_A,
            typename T_b, typename T_B, typename I_B, MatrixView_TArgs targs_B,
                          typename T_C, typename I_C, MatrixView_TArgs targs_C
        >
        void geadd(
            T_a alpha, cref<MatrixView<T_A,I_A,targs_A>> A,
            T_b beta,  cref<MatrixView<T_B,I_B,targs_B>> B,
                       mref<MatrixView<T_C,I_C,targs_C>> C
        )
        {
            // TODO: Implement this.
            
            // TODO: Allow B == C or A == C?
            
            // TODO: Check overlap?
        }
        
        
    } namespace Dense
    
} // namespace Tensors
