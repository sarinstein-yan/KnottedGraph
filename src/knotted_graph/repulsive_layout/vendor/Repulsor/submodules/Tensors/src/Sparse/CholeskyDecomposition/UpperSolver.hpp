namespace Tensors
{
    namespace Sparse
    {
        
        template<typename Scal_, IntQ Int_, IntQ LInt_, Parallel_T parQ>
        class CholeskyDecomposition;
        
        template<bool mult_rhs, typename Scal_, IntQ Int_, IntQ LInt_>
        class alignas(ObjectAlignment) UpperSolver final
        {
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
        protected:
            
            static constexpr Int izero = 0;
            static constexpr Int ione  = 1;
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;

            const Int nrhs    = 0;
            const Int max_n_1 = 0;
            
            cptr<Int>  SN_rp;
            cptr<LInt> SN_outer;
            cptr<Int>  SN_inner;
            
            cptr<LInt> SN_tri_ptr;
            mptr<Scal> SN_tri_val;
            cptr<LInt> SN_rec_ptr;
            mptr<Scal> SN_rec_val;
            
            // On input: the right hand side; on output: the solution.
            mptr<Scal> X;
            
            // Working space for BLAS3 routines.
            Tensor1<Scal,Int> X_1_buffer;
            
            // X_1 is the part of X that interacts with U_1, size = n_0 x n_1.
            mptr<Scal> X_1;
            
            // x_1 is the part of x that interacts with U_1, size = n_1.
            mptr<Scal> x_1;
            
        public:
            
            // No default constructor
            UpperSolver() = delete;
            // Destructor
            ~UpperSolver() = default;
            // Copy constructor
            UpperSolver( const UpperSolver & other ) = default;
            // Copy assignment operator
            UpperSolver & operator=( const UpperSolver & other ) = default;
            // Move constructor
            UpperSolver( UpperSolver && other ) = default;
            // Move assignment operator
            UpperSolver & operator=( UpperSolver && other ) = default;

//            /*!@brief Initialize solver with external numerical factorization.
//             */
//            
//            UpperSolver(
//                CholeskyDecomposition<Scal,Int,LInt> & chol,
//                mptr<Scal> SN_tri_val_,
//                mptr<Scal> SN_rec_val_,
//                mptr<Scal> X_,
//                const Int nrhs_
//            )
//            :   nrhs            ( nrhs_                  )
//            ,   max_n_1         ( chol.max_n_1           )
//            ,   SN_rp           ( chol.SN_rp.data()      )
//            ,   SN_outer        ( chol.SN_outer.data()   )
//            ,   SN_inner        ( chol.SN_inner.data()   )
//            ,   SN_tri_ptr      ( chol.SN_tri_ptr.data() )
//            ,   SN_tri_val      ( SN_tri_val_            )
//            ,   SN_rec_ptr      ( chol.SN_rec_ptr.data() )
//            ,   SN_rec_val      ( SN_rec_val_            )
//            ,   X               ( X_                     )
//            ,   X_1_buffer      ( max_n_1 * nrhs         )
//            ,   X_1             ( X_1_buffer.data()      )
//            ,   x_1             ( X_1_buffer.data()      )
//            {}
//            
//            /*!@brief Initialize solver with internal numerical factorization.
//             */
//            
//            UpperSolver(
//                CholeskyDecomposition<Scal,Int,LInt> & chol, const Int nrhs_
//            )
//            : UpperSolver(
//                   chol,
//                   chol.SN_tri_val.data(),
//                   chol.SN_rec_val.data(),
//                   chol.X.data()
//              )
//            {}
            
            template<Parallel_T parQ>
            UpperSolver(
                CholeskyDecomposition<Scal,Int,LInt,parQ> & chol, const Int nrhs_
            )
            :   nrhs            ( nrhs_                         )
            ,   max_n_1         ( chol.max_n_1                  )
            ,   SN_rp           ( chol.SN_rp.data()             )
            ,   SN_outer        ( chol.SN_outer.data()          )
            ,   SN_inner        ( chol.SN_inner.data()          )
            ,   SN_tri_ptr      ( chol.SN_tri_ptr.data()        )
            ,   SN_tri_val      ( chol.SN_data.tri_val.data()   )
            ,   SN_rec_ptr      ( chol.SN_rec_ptr.data()        )
            ,   SN_rec_val      ( chol.SN_data.rec_val.data()   )
            ,   X               ( chol.X.data()                 )
            ,   X_1_buffer      ( max_n_1 * nrhs                )
            ,   X_1             ( X_1_buffer.data()             )
            ,   x_1             ( X_1_buffer.data()             )
            {}
            
        public:
                
            // Solver routine.
            void operator()( const Int s )
            {   
                const Int n_0 = SN_rp[s+1] - SN_rp[s];
            
                assert_positive(n_0);
                
                const LInt l_begin = SN_outer[s  ];
                const LInt l_end   = SN_outer[s+1];
                
                const Int n_1 = int_cast<Int>(l_end - l_begin);
                
                // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                cptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[s]];
                
                // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                cptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[s]];

                
                if constexpr ( mult_rhs )
                {
                    // X_0 is the part of X that interacts with U_0, size = n_0 x nrhs.
                    mptr<Scal> X_0 = &X[nrhs * SN_rp[s]];
                    
                    // Load the already computed values into X_1.
                    for( Int j = 0; j < n_1; ++j )
                    {
                        copy_buffer( &X[nrhs * SN_inner[l_begin+j]], &X_1[nrhs * j], nrhs );
                    }

                    if( n_0 == ione )
                    {
                        if( n_1 > izero )
                        {
                            // Compute X_0 -= U_1 * X_1

                            //  U_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                            //  X_1 is a matrix of size n_1 x nrhs.
                            //  X_0 is a matrix of size 1 x nrhs; we can interpret it as vector of size nrhs.

                            // Hence, we can compute X_0^T -= X_1^T * U_1^T via gemv instead:
                            BLAS::gemv<Layout::RowMajor,Op::Trans>(
                                n_1, nrhs,
                                -one, X_1, nrhs,
                                      U_1, ione,        // XXX Problem: We need Conj(U_1)!
                                 one, X_0, ione
                            );
                        }

                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        
                        scale_buffer( Inv(U_0[0]), X_0, nrhs );
                    }
                    else // using BLAS3 routines.
                    {
                        if( n_1 > izero )
                        {
                            // Compute X_0 -= U_1 * X_1
                            BLAS::gemm<Layout::RowMajor,Op::Id,Op::Id>(
                                // XX Op::Id -> Op::ConjugateTranspose
                                n_0, nrhs, n_1,
                                -one, U_1, n_1,      // XXX n_1 -> n_0
                                      X_1, nrhs,
                                 one, X_0, nrhs
                            );
                        }
                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        BLAS::trsm<Layout::RowMajor,Side::Left,UpLo::Upper,Op::Id,Diag::NonUnit>(
                            n_0, nrhs,
                            one, U_0, n_0,
                                 X_0, nrhs
                        );
                    }
                }
                else // mult_rhs == false
                {
                    // x_0 is the part of x that interacts with U_0, size = n_0.
                    mptr<Scal> x_0 = &X[SN_rp[s]];

                    if( n_0 == one )
                    {
                        Scal U_1x_1 = 0;

                        if( n_1 > izero )
                        {
                            // Compute X_0 -= U_1 * X_1
                            //  U_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                            //  x_1 is a vector of size n_1.
                            //  x_0 is a matrix of size 1 x 1; we can interpret it as vector of size 1.

                            // Hence, we can compute X_0 -= U_1 * X_1 via a simple dot product.
                            // Beware: Scattered read.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                U_1x_1 += U_1[j] * X[SN_inner[l_begin+j]];
                            }
                            
                        }

                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        x_0[0] = (x_0[0] - U_1x_1) / U_0[0];
                    }
                    else // n_0 > one; using BLAS2 routines.
                    {
                        if( n_1 > izero )
                        {
                            // Load the already computed values into x_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                x_1[j] = X[SN_inner[l_begin+j]];
                            }

                            // Compute x_0 -= U_1 * x_1
                            BLAS::gemv<Layout::RowMajor,Op::Id>(// XXX Op::Id -> Op::ConjTrans
                                n_0, n_1,
                                -one, U_1, n_1, // XXX n_1 -> n_0
                                      x_1, ione,
                                 one, x_0, ione
                            );
                        }

                        // Triangle solve U_0 * x_0 = B while overwriting x_0.
                        BLAS::trsv<Layout::RowMajor,UpLo::Upper,Op::Id,Diag::NonUnit>(
                            n_0, U_0, n_0, x_0, ione
                        );
                    }
                }
            }

    
        public:
            
            std::string ClassName() const
            {
                return std::string("Sparse::UpperSolver")+"<"+ToString(mult_rhs)+
                ","+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // class UpperSolve
        
        
    } // namespace Sparse
    
} // namespace Tensors

