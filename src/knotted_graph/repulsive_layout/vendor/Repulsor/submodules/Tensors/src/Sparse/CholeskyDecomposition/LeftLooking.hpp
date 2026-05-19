#pragma once

// TODO: Would be great to allow a CholeskyFactorizer_LeftLooking to use more than just one thread, e.g. if it is at the top of the tree...

namespace Tensors
{
    namespace Sparse
    {
        
        template<typename Scal_, IntQ Int_, IntQ LInt_, Parallel_T parQ>
        class CholeskyDecomposition;
        
        template<typename Scal_, IntQ Int_, IntQ LInt_>
        class alignas(ObjectAlignment) CholeskyFactorizer_LeftLooking final
        {
            // Performs a left-looking factorization.
            
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
        protected:
            
            static constexpr bool debug = false;
            
            static constexpr Int izero = 0;
            static constexpr Int ione  = 1;
            
            static constexpr Scal zero = 0;
            static constexpr Scal one  = 1;


            const Int n;
            
            // shared data
            cptr<LInt>       A_diag; // position to the diagonal element of A in each row.
            cptr<LInt>       A_rp;   // row pointers of upper triangle of A
            cptr<Int>        A_ci;   // column indices of upper triangle of A
            cptr<Scal>       A_val;  // values of upper triangle of A
            
            Scal             reg;     // Regularization parameter for the diagonal.
            
            cptr<Int>        child_ptr;
            cptr<Int>        child_idx;
            
            cptr<Int>        SN_rp;
            cptr<LInt>       SN_outer;
            cptr<Int>        SN_inner;
            
            cptr<LInt>       SN_tri_ptr;
            mptr<Scal>       SN_tri_val;
            cptr<LInt>       SN_rec_ptr;
            mptr<Scal>       SN_rec_val;
            
            cptr<Int>        desc_counts;
            
            
            // local data
            
            // Working space for intersection calculations.
            Tensor1<Int,Int> II_pos_buffer;
            Tensor1<Int,Int> IL_pos_buffer;
            Tensor1<Int,Int> JJ_pos_buffer;
            Tensor1<Int,Int> JL_pos_buffer;
            
            mptr<Int> II_pos = nullptr;
            mptr<Int> IL_pos = nullptr;
            mptr<Int> JJ_pos = nullptr;
            mptr<Int> JL_pos = nullptr;
            
            Int IL_len = 0;
            Int JL_len = 0;
            
            // Working space for BLAS3 routines.
            Tensor1<Scal,Int> B_0_buffer; // part that updates U_0
            Tensor1<Scal,Int> B_1_buffer; // part that updates U_1
            Tensor1<Scal,Int> C_0_buffer; // scattered subblock of U_0
            Tensor1<Scal,Int> C_1_buffer; // scattered subblock of U_1
            
            mptr<Scal> B_0 = nullptr;
            mptr<Scal> B_1 = nullptr;
            mptr<Scal> C_0 = nullptr;
            mptr<Scal> C_1 = nullptr;

            bool goodQ = true;
            
            // Monitors.
            Int intersec = 0;
            Int empty_intersec_undetected = 0;
            Int empty_intersec_detected   = 0;
            
            Int IL_len_small = 0;
            Int JL_len_small = 0;
            Int IL_len_and_JL_len_small = 0;

            
            Int  intersection_counter = 0;
            Int  empty_intersection_counter = 0;
            
            float fetch_from_decendants_time = 0;
            float intersection_time = 0;
            float empty_intersection_time = 0;
            float sup_sup_time = 0;
            float col_sup_time = 0;
            float scatter_time = 0;
            
            float fetch_from_A_time          = 0;
            float factorize_supernode_time   = 0;
            
        public:
            
            // No default constructor
            CholeskyFactorizer_LeftLooking() = delete;
            // Destructor
            ~CholeskyFactorizer_LeftLooking() = default;
            // Copy constructor
            CholeskyFactorizer_LeftLooking( const CholeskyFactorizer_LeftLooking & other ) = default;
            // Copy assignment operator
            CholeskyFactorizer_LeftLooking & operator=( const CholeskyFactorizer_LeftLooking & other ) = default;
            // Move constructor
            CholeskyFactorizer_LeftLooking( CholeskyFactorizer_LeftLooking && other ) = default;
            // Move assignment operator
            CholeskyFactorizer_LeftLooking & operator=( CholeskyFactorizer_LeftLooking && other ) = default;
            
            
            template<Parallel_T parQ>
            CholeskyFactorizer_LeftLooking( CholeskyDecomposition<Scal,Int,LInt,parQ> & chol )
            // shared data
            :   n               ( chol.n                                        )
            ,   A_diag          ( chol.A.Diag().data()                          )
            ,   A_rp            ( chol.A.Outer().data()                         )
            ,   A_ci            ( chol.A.Inner().data()                         )
            ,   A_val           ( chol.A_val.data()                             )
            ,   reg             ( chol.reg                                      )
            ,   child_ptr       ( chol.AssemblyTree().ChildPointers().data()    )
            ,   child_idx       ( chol.AssemblyTree().ChildIndices().data()     )
            ,   SN_rp           ( chol.SN_rp.data()                             )
            ,   SN_outer        ( chol.SN_outer.data()                          )
            ,   SN_inner        ( chol.SN_inner.data()                          )
            ,   SN_tri_ptr      ( chol.SN_tri_ptr.data()                        )
            ,   SN_tri_val      ( chol.SN_data.tri_val.data()                   )
            ,   SN_rec_ptr      ( chol.SN_rec_ptr.data()                        )
            ,   SN_rec_val      ( chol.SN_data.rec_val.data()                   )
            ,   desc_counts     ( chol.AssemblyTree().DescendantCounts().data() )
            
            // local data
            ,   II_pos_buffer   ( chol.max_n_0                                  )
            ,   IL_pos_buffer   ( chol.max_n_0                                  )
            ,   JJ_pos_buffer   ( chol.max_n_1                                  )
            ,   JL_pos_buffer   ( chol.max_n_1                                  )
            ,   II_pos          ( II_pos_buffer.data()                          )
            ,   IL_pos          ( IL_pos_buffer.data()                          )
            ,   JJ_pos          ( JJ_pos_buffer.data()                          )
            ,   JL_pos          ( JL_pos_buffer.data()                          )
            ,   B_0_buffer      ( chol.max_n_0 * chol.max_n_0                   )
            ,   B_1_buffer      ( chol.max_n_0 * chol.max_n_1                   )
            ,   C_0_buffer      ( chol.max_n_0 * chol.max_n_0                   )
            ,   C_1_buffer      ( chol.max_n_0 * chol.max_n_1                   )
            ,   B_0             ( B_0_buffer.data()                             )
            ,   B_1             ( B_1_buffer.data()                             )
            ,   C_0             ( C_0_buffer.data()                             )
            ,   C_1             ( C_1_buffer.data()                             )
            {}
                        
        public:
                
            // Factorization routine.
            void operator()( const Int s )
            {
                const  Int i_begin = SN_rp[s  ];
                const  Int i_end   = SN_rp[s+1];
                
                const LInt l_begin = SN_outer[s  ];
                const LInt l_end   = SN_outer[s+1];
                
                const  Int n_0     = i_end - i_begin;
                const  Int n_1     = int_cast<Int>(l_end - l_begin);
                
                assert_positive(n_0);
                
                if( n_0 <= izero )
                {
                    eprint(ClassName()+"::operator(): n_0<=0");
                    TOOLS_DUMP(s);
                    TOOLS_DUMP(SN_rp[s  ]);
                    TOOLS_DUMP(SN_rp[s+1]);
                }
                
                // U_0 is interpreted as an upper triangular matrix of size n_0 x n_0.
                // U_1 is interpreted as a  rectangular      matrix of size n_0 x n_1.
                mptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[s]];
                mptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[s]];
                
                FetchFromA( i_begin, i_end, l_begin, l_end, U_0, U_1);
                
                // We have to fetch the row updates of all descendants of s.
                // We assume that the descendants of s are already factorized.
                
                // Since aTree is postordered, we can exploit that all descendants of s lie
                // contiguously in memory directly before s.
                
                const Int t_begin = s - desc_counts[s];
                const Int t_end   = s;
                
                // TODO: This could be parallelized by adding into local buffers V_0 and V_1 and reducing them into U_0 and U_1.

                FetchFromDescendants( s, t_begin, t_end, n_0, n_1, U_0, U_1 );
                
                FactorizeSupernode( n_0, n_1, U_0, U_1 );
            }
            
        protected:
            
            void FetchFromA(
                Int  i_begin,  Int i_end,
                LInt l_begin, LInt l_end,
                mptr<Scal> U_0, mptr<Scal> U_1
            )
            {
//                const Time start_time = Clock::now();
                
                // Read the values of A into U_0 and U_1.

                const  Int n_0 = i_end - i_begin;
                const  Int n_1 = int_cast<Int>(l_end - l_begin);
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt k_begin = A_diag[i  ];
                    const LInt k_end   = A_rp  [i+1];
                    
                    LInt k;
                    
                    {
                        k = k_begin;
                        
                        const Int j = A_ci [k];
                        
                        U_0[n_0 * (i-i_begin) + (j-i_begin)] = static_cast<Scal>(A_val[k]+reg);
                        
                        ++k;
                    }
                    for( ; k < k_end; ++k )
                    {
                        const Int j = A_ci [k];
                        
                        if( j < i_end )
                        {
                            // j belongs to the triangular part
                            U_0[n_0 * (i-i_begin) + (j-i_begin)] = static_cast<Scal>(A_val[k]);
                        }
                        else
                        {
                            break;
                        }
                    }
                    
                    if( k >= k_end )
                    {
                        continue;
                    }
                    
                    
                    // If we arrive here, then there are still a few entries of A to be sorted in.
                    // From now on the insertion position must be in the rectangular part.
                    
                    LInt l    = l_begin;
                    Int col_l = SN_inner[l];
                    
                    // Continue the loop where the previous one was aborted.
                    for( ; k < k_end; ++k )
                    {
                        const Int j = A_ci [k];
                        
                        //TODO: This could _maybe_ a bit faster by using a lookup table for indices.
                        
                        // Find the position l of j in SN_inner, then write into U_1[l - l_begin].
                        // Remark: We don't need a check l < l_end here, because we know that we find an l.
                        while( col_l < j )
                        {
                            col_l = SN_inner[++l];
                        }
                        U_1[n_1 * (i - i_begin) + (l - l_begin)] = static_cast<Scal>(A_val[k]);
                    }
                }
                
//                fetch_from_A_time +=  Tools::Duration( start_time, Clock::now() );
            }
            
            void FetchFromDescendants(
                const Int s,                            // the supernode into which to fetch
                const Int t_begin, const Int t_end,     // the range of descendants
                const Int n_0, const Int n_1, mptr<Scal> U_0, mptr<Scal> U_1
            )
            {
//                const Time start_time = Clock::now();
                
                // Incorporate the row updates from descendants [ t_begin,...,t_end [ into U_0 and U_1.
                
//                for( Int t = t_begin; t < t_end; ++t )
                for( Int t = t_end; t --> t_begin; )
                {
                    // Compute the intersection of supernode s with t.
                    
//                    Time intersection_start_time = Clock::now();
                    
                    ComputeIntersection( s, t );
                    
//                    CheckIntersection( s, t );
                    
                    if( (IL_len <= 0) && (JL_len <= 0) )
                    {
                        ++empty_intersection_counter;
                        
                        // Observation (yet to be proven):
                        // If t is a descendant of s that does not contribute to s,
                        // then all of t's descendants also don't contribute to s!
                        
                        // So we can jump over all descendants of t.
                        
                        t = t - desc_counts[t];
                        
//                        empty_intersection_time += Tools::Duration( intersection_start_time, Clock::now() );
                        
                        continue;
                    }
                    else
                    {
//                        intersection_time += Tools::Duration( intersection_start_time, Clock::now() );
                    }
                    
                    const Int m_0 = SN_rp[t+1] - SN_rp[t];
                    const Int m_1 = int_cast<Int>(SN_outer[t+1] - SN_outer[t]);

                    cptr<Scal> t_rec = &SN_rec_val[SN_rec_ptr[t]];
                    // t_rec is interpreted as a rectangular matrix of size m_0 x m_1.

                    // TODO: Maybe we should transpose U_0 and U_1 etc. to reduce amount of scattered-reads and adds...
                    // TODO: Then U_0 and U_1 would be ColMajor and we could use BLAS without the C-layer CBLAS.
                    
                    if( m_0 > ione )
                    {
//                        Time sup_sup_start_time = Clock::now();
                        
                        // Update triangular block U_0.
                        if( IL_len > izero )
                        {
                            // Col-scatter-read t_rec[:,IL_pos] into B_0,
                            // where B_0 is a matrix of size m_0 x IL_len;
                            
                            for( Int i = 0; i < m_0; ++i )
                            {
                                scatter_read( &t_rec[m_1 * i], &B_0[IL_len * i], IL_pos, IL_len );
                            }
                            
    //                        for( Int i = 0; i < IL_len; ++i )
    //                        {
    //                            copy_buffer( &t_rec[m_0 * IL_pos[i]], &B_0[m_0 * i], m_0 );
    //                        }
                            
                            // Do C_0 = - upper(B_0^H * B_0),
                            // where C_0 is an upper triangular matrix of size IL_len x IL_len.

                            BLAS::herk<Layout::RowMajor,UpLo::Upper,Op::ConjTrans>(
                                IL_len, m_0,
                                -Scalar::Real<Scal>(1), B_0, IL_len,
                                 Scalar::Real<Scal>(0), C_0, IL_len
                            );

                            // Row-col-scatter-add C_0 into U_0,
                            // where U_0 is an upper triangular matrix of size n_0 x n_0.
                            
//                            Time scatter_start_time = Clock::now();
                            
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                for( Int j = i; j < IL_len; ++j )
                                {
                                    U_0[ n_0 * II_pos[i] + II_pos[j] ] += C_0[ IL_len * i + j ];
                                }
                            }
                            
//                            scatter_time += Tools::Duration( scatter_start_time, Clock::now() );
                            
                        } // if( IL_len > izero )

                        // Update rectangular block U_1.
                        if( (IL_len > izero) && (JL_len > izero) )
                        {
                            // Col-scatter-read t_rec[:,JL_pos] from B_1,
                            // where B_1 is a matrix of size m_0 x JL_len.
                            
                            for( Int i = 0; i < m_0; ++i )
                            {
                                scatter_read( &t_rec[m_1 * i], &B_1[JL_len * i], JL_pos, JL_len );
                            }
                            
                            // Do C_1 = - B_0^H * B_1,
                            // where
                            //          B_0 is a matrix of size m_0    x IL_len,
                            //          B_1 is a matrix of size m_0    x JL_len,
                            //          C_1 is a matrix of size IL_len x JL_len.

                            
                            if( JL_len > ione )
                            {
                                if( IL_len > ione )
                                {
                                    // IL_len > ione and JL_len > ione: Use BLAS 3

                                    BLAS::gemm<Layout::RowMajor,Op::ConjTrans,Op::Id>(
                                        IL_len, JL_len, m_0,
                                        -one, B_0, IL_len,
                                              B_1, JL_len,
                                        zero, C_1, JL_len
                                    );
                                }
                                else // IL_len == ione
                                {
                                    // IL_len == ione and JL_len > ione: Use BLAS 2
                                    //
                                    // Do C_1 = - B_0^H * B_1,
                                    // where
                                    //          B_0^H is a matrix of size   1 x m_0,
                                    //          B_1 is a matrix of size   m_0 x JL_len,
                                    //          C_1 is a matrix of size     1 x JL_len.
                                    //
                                    // This is equivalent to
                                    //
                                    // C_1^T  = - B_1^T * conj(B_0)
                                    // where
                                    //          B_0 is a matrix of size      m_0 x 1,
                                    //          B_1^T is a matrix of size JL_len x m_0,
                                    //          C_1^T is a matrix of size JL_len x 1.
                                    
                                    
                                    // TODO: Conjugate B_0 while scatter-reading.
                                    // TODO: This would remove the need for gemm.
                                    if constexpr ( Scalar::RealQ<Scal> )
                                    {
                                        BLAS::gemv<Layout::RowMajor,Op::Trans>(
                                            m_0, JL_len,
                                            -one, B_1, JL_len,
                                                  B_0, ione,
                                            zero, C_1, ione
                                        );
                                    }
                                    else
                                    {
                                        // We use gemm because gemv does not allow us to conjugate B_0.
                                        
                                        BLAS::gemm<Layout::RowMajor,Op::ConjTrans,Op::Id>(
                                            ione, JL_len, m_0,
                                            -one, B_0, ione,
                                                  B_1, JL_len,
                                            zero, C_1, JL_len
                                        );
                                    }
                                }
                                
                                
                            }
                            else // JL_len == ione -- But this specialization does not seem to make a big difference.
                            {
                                // Do C_1 = - B_0^H * B_1,
                                // where C_1 is a matrix of size IL_len x 1.
                                
                                if( IL_len > ione )
                                {
                                    // IL_len > ione and JL_len == ione: Use BLAS 2
                                    
                                    BLAS::gemv<Layout::RowMajor,Op::ConjTrans>(
                                        m_0, IL_len,
                                        -one, B_0, IL_len,
                                              B_1, ione,
                                        zero, C_1, ione
                                    );
                                }
                                else // IL_len == ione
                                {
                                    // IL_len == ione and JL_len == ione: Use BLAS 1
                                    
                                    C_1[0] = -dot_buffers<VarSize,Sequential,Op::Conj,Op::Id>(
                                        B_0, B_1, m_0
                                    );
                                }
                            }
                            
                            // Row-col-scatter-add C_1 into U_1,
                            // where U_1 is a matrix of size n_0 x n_1.
                            
//                            Time scatter_start_time = Clock::now();
                            
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                mptr<Scal> U_1_i = &U_1[n_1 * II_pos[i]];
                                cptr<Scal> C_1_i = &C_1[JL_len * i];
                                
                                for( Int j = 0; j < JL_len; ++j )
                                {
                                    U_1_i[JJ_pos[j]] += C_1_i[j];
                                }
                            }
                            
//                            scatter_time += Tools::Duration( scatter_start_time, Clock::now() );
                            
                        } // if( (IL_len > izero) && (JL_len > izero) )

//                        sup_sup_time += Tools::Duration( sup_sup_start_time, Clock::now() );
                    }
                    else // m_0 == ione
                    {
//                        Time col_sup_start_time = Clock::now();
                        
                        // In this case we have to form only (scattered) outer products of vectors, which are basically BLAS2 routines. Unfortunately, ?ger, ?syr, ?her, etc. add into the target matrix without the option to zero it out first. Hence, we simply write these double loops ourselves. Since this is BLAS2 and not BLAS3, and since IL_len and JL_len are often not particularly long, there isn't that much room for optimization anyways. And since we cannot use ?ger / ?her, we fuse the scattered read/write operations directly into these loops.
                        
                        for( Int j = 0; j < IL_len; ++j )
                        {
                            B_0[j] = t_rec[IL_pos[j]];
                        }
                        
                        if( JL_len > izero )
                        {
                            for( Int j = 0; j < JL_len; ++j )
                            {
                                B_1[j] = t_rec[JL_pos[j]];
                            }
                            
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                const Scal factor = - Conj(t_rec[IL_pos[i]]);

                                const Int i_ = II_pos[i];
                                mptr<Scal> U_0_i = &U_0[n_0 * i_];
                                mptr<Scal> U_1_i = &U_1[n_1 * i_];
                                
                                for( Int j = i; j < IL_len; ++j )
                                {
                                    U_0_i[II_pos[j]] += factor * B_0[j];
                                }

                                for( Int j = 0; j < JL_len; ++j )
                                {
                                    U_1_i[JJ_pos[j]] += factor * B_1[j];
                                }
                            }
                        }
                        else
                        {
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                const Scal factor = - Conj(t_rec[IL_pos[i]]);

                                const Int i_ = II_pos[i];
                                mptr<Scal> U_0_i = &U_0[n_0 * i_];

                                for( Int j = i; j < IL_len; ++j )
                                {
                                    U_0_i[II_pos[j]] += factor * B_0[j];
                                }
                            }
                        }
                        
//                        col_sup_time += Tools::Duration( col_sup_start_time, Clock::now() );
                        
                    } // if( m_0 > ione )
                        
                    
                } // for( Int t = t_begin; t < t_end; ++t )
                
//                fetch_from_decendants_time += Tools::Duration( start_time, Clock::now() );
            }
            
            void FactorizeSupernode( Int n_0, Int n_1, mptr<Scal> U_0, mptr<Scal> U_1 )
            {
//                Time start_time =  Clock::now();
                
                if( n_0 > ione )
                {
                    // Cholesky factorization of U_0
                    const int info = LAPACK::potrf<Layout::RowMajor,UpLo::Upper>( n_0, U_0, n_0);
                    
                    goodQ = goodQ && (info == 0);

                    // Triangular solve U_1 = U_0^{-H} U_1.
                    if( n_1 > ione )
                    {
                        BLAS::trsm<Layout::RowMajor,Side::Left,UpLo::Upper,Op::ConjTrans,Diag::NonUnit>(
                            n_0, n_1, one, U_0, n_0, U_1, n_1
                        );
                    }
                    else if( n_1 == ione )
                    {
                        BLAS::trsv<Layout::RowMajor,UpLo::Upper,Op::ConjTrans,Diag::NonUnit>(
                            n_0, U_0, n_0, U_1, ione
                        );
                    }
                    else // n_1 == izero
                    {
                        // Do nothing.
                    }
                }
                else
                {
                    U_0[0] = Sqrt(Abs(U_0[0]));
                    scale_buffer(static_cast<Scal>(1)/U_0[0], U_1, n_1);
                }
                
//                factorize_supernode_time += Tools::Duration( start_time, Clock::now() );
            }
            
        protected:
            
            TOOLS_FORCE_INLINE void scatter_read( cptr<Scal> x, mptr<Scal> y, cptr<Int> idx, Int N )
            {
//                Time start_time = Clock::now();
                
                for( ; N --> izero; ) { y[N] = x[idx[N]]; }
                
//                scatter_time += Tools::Duration( start_time, Clock::now() );
            }
            
            TOOLS_FORCE_INLINE void scatter_add( cptr<Scal> x, mptr<Scal> y, cptr<Int> idx, Int N )
            {
                for( ; N --> izero; ) { y[idx[N]] += x[N]; }
            }
            

            void ComputeIntersection( const Int s, const Int t )
            {
                ++intersection_counter;
                // Compute the intersecting column indices of s-th and t-th supernode.
                // We assume that hence that t is a descendant of s.
                // In particular, this implies t < s
                
                // s-th supernode has triangular part
                //
                //  I = [SN_rp[s],SN_rp[s]+1,...,SN_rp[s+1][
                //
                // and rectangular part
                //
                //  J = [SN_inner[SN_outer[s]],[SN_inner[SN_outer[s]+1],...,[
                //
                // t-th supernode has triangular part
                //
                //  K = [SN_rp[t],SN_rp[t]+1,...,SN_rp[t+1][
                //
                // and rectangular part 
                //
                //  L = [SN_inner[SN_outer[t]],[SN_inner[SN_outer[t]+1],...,[
                //
                
                // We have to compute
                // - the positions II_pos of I \cap L in I,
                // - the positions IL_pos of I \cap L in L,
                // - the positions JJ_pos of J \cap L in J,
                // - the positions JL_pos of J \cap L in L.

                // On return the numbers IL_len, JL_len contain the lengths of the respective lists.
  

                IL_len = 0;
                JL_len = 0;
                
                ++intersec;
                
                const  Int i_begin = SN_rp[s  ];
                const LInt l_end   = SN_outer[t+1];
                
                
                // Quick check whether the convex hulls of index ranges overlap.
                if( SN_inner[l_end-1] < i_begin )
                {
//                    ++empty_intersec_detected;
                    return;
                }
                
                // Go through I and L in ascending order and collect intersection indices.

                const  Int i_end   = SN_rp[s+1];
                
                const LInt j_begin = SN_outer[s  ];
                const LInt j_end   = SN_outer[s+1];
                
                const LInt l_begin = SN_outer[t  ];
                
                TOOLS_DEBUG_ASSERT( l_begin < l_end, "We should have l_begin < l_end; otherwise t would not be a descendant of s, right?" ); // bc. s would not be a child of t otherwise.
                
                 Int i = i_begin;
                LInt l = l_begin;
                
                Int L_l = SN_inner[l];
                
                while( true )
                {
                    if( i < L_l )
                    {
                        ++i;
                        
                        if( i >= i_end )
                        {
                            break;
                        }
                    }
                    else if( L_l < i )
                    {
                        ++l;
                        
                        if( l >= l_end )
                        {
                            return;
                        }
                        else
                        {
                            L_l = SN_inner[l];
                        }
                    }
                    else // i == L_l
                    {
                        II_pos[IL_len] = int_cast<Int>(i-i_begin);
                        IL_pos[IL_len] = int_cast<Int>(l-l_begin);
                        ++IL_len;
                        ++i;
                        ++l;
                        
                        if( l >= l_end )
                        {
                            return;
                        }
                        else
                        {
                            L_l = SN_inner[l];
                        }
                        
                        if( i >= i_end )
                        {
                            break;
                        }
  
                    }
                }
                
                // Go through J and L in ascending order and collect intersection indices.
                
                if( j_begin == j_end )
                {
                    return;
                }
                
                LInt j = j_begin;
//                LInt l = l_begin;         // We can continue with l where it were before...
                
                Int J_j = SN_inner[j];
//                Int L_l = SN_inner[l];    // ... and thus, we can keep the old L_l, too.
                
                while( true )
                {
                    if( J_j < L_l )
                    {
                        ++j;
                        
                        if( j < j_end )
                        {
                            J_j = SN_inner[j];
                        }
                        else
                        {
                            return;
                        }
                    }
                    else if( L_l < J_j )
                    {
                        ++l;
                        
                        if( l < l_end )
                        {
                            L_l = SN_inner[l];
                        }
                        else
                        {
                            return;
                        }
                    }
                    else // J_j == L_l
                    {
                        JJ_pos[JL_len] = int_cast<Int>(j-j_begin);
                        JL_pos[JL_len] = int_cast<Int>(l-l_begin);
                        ++JL_len;
                        ++j;
                        ++l;
                        
                        if( (l < l_end) && (j < j_end) )
                        {
                            L_l = SN_inner[l];
                            J_j = SN_inner[j];
                        }
                        else
                        {
                            return;
                        }
                    }
                }
                
//                if( (IL_len == izero) && (JL_len == izero) )
//                {
//                    ++empty_intersec_undetected;
//                }
            }
            
            // For debugging reasons only.
            void PrintIntersection( const Int s, const Int t )
            {
                print("");
                
                TOOLS_DUMP(s);
                TOOLS_DUMP(t);

                //I = [SN_rp[s],SN_rp[s]+1,...,SN_rp[s+1][
                
                Tensor1<Int,Int> I ( SN_rp[s+1] - SN_rp[s] );
                fill_range_buffer( I.data(), SN_rp[s], SN_rp[s+1] - SN_rp[s] );
                
                Tensor1<Int,Int> J ( &SN_inner[SN_outer[s]], SN_outer[s+1] - SN_outer[s] );
                
                Tensor1<Int,Int> K ( SN_rp[t+1] - SN_rp[t] );
                fill_range_buffer( K.data(), SN_rp[t], SN_rp[t+1] - SN_rp[t] );
                
                Tensor1<Int,Int> L ( &SN_inner[SN_outer[t]], SN_outer[t+1] - SN_outer[t] );

                Tensor1<Int,Int> I_cap_L_0( IL_len );
                Tensor1<Int,Int> I_cap_L_1( IL_len );
                Tensor1<Int,Int> J_cap_L_0( JL_len );
                Tensor1<Int,Int> J_cap_L_1( JL_len );
                
                for( Int idx = 0; idx < IL_len; ++idx )
                {
                    I_cap_L_0[idx] = I[ II_pos[idx] ];
                    I_cap_L_1[idx] = L[ IL_pos[idx] ];
                }
                
                for( Int idx = 0; idx < JL_len; ++idx )
                {
                    I_cap_L_0[idx] = J[ JJ_pos[idx] ];
                    I_cap_L_1[idx] = L[ JL_pos[idx] ];
                }
                
                TOOLS_DUMP(I);
                TOOLS_DUMP(J);
                TOOLS_DUMP(K);
                TOOLS_DUMP(L);
                
                TOOLS_DUMP(I_cap_L_0);
                TOOLS_DUMP(I_cap_L_1);
                TOOLS_DUMP(J_cap_L_0);
                TOOLS_DUMP(J_cap_L_1);
                
                print("");
            }
            
            
            void CheckIntersection( const Int s, const Int t ) const
            {
                // Beware: This test only checks whether all _found_ pairs match!
                // Beware: It does not check, whether _all_ pairs are found!
                bool okay = true;

                const  Int i_begin = SN_rp[s  ];
                const  Int i_end   = SN_rp[s+1];
                
                const  Int j_begin = SN_outer[s  ];
                const  Int j_end   = SN_outer[s+1];
                
                const LInt l_begin = SN_outer[t  ];
                const LInt l_end   = SN_outer[t+1];

                const Int n_0 = i_end - i_begin;
                const Int n_1 = j_end - j_begin;
                
                const Int m_1 = l_end - l_begin;
                
                for( Int i = 0; i < IL_len; ++i )
                {
                    okay = okay && ( i_begin + II_pos[i] == SN_inner[ l_begin + IL_pos[i] ] );
                }
                
                if( !okay )
                {
                    eprint("bug in II_pos + IL_pos");
                    
                    TOOLS_DUMP(n_0);
                    TOOLS_DUMP(m_1);
                    
                    Tensor1<Int,Int> a (n_0);
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        a[i-i_begin] = i;
                    }
                    
//                    valprint( "I", ToString( &a[0], n_0 , 16 ) );
//                    valprint( "L", ToString( &SN_inner[l_begin], m_1 , 16 ) );
//
//                    valprint( "II_pos", ToString(II_pos,IL_len,16) );
//                    valprint( "IL_pos", ToString(IL_pos,IL_len,16) );
                }
                
                for( Int j = 0; j < JL_len; ++j )
                {
                    okay = okay && ( SN_inner[j_begin + JJ_pos[j]] == SN_inner[l_begin + JL_pos[j]] );
                }
                
                if( !okay )
                {
                    eprint("bug in JJ_pos + JL_pos");
                    
                    TOOLS_DUMP(n_1);
                    TOOLS_DUMP(m_1);
                }
            }
            
        public:
            
            bool GoodQ() const
            {
                return goodQ;
            }
            
            std::string ClassName() const
            {
                return std::string("Sparse::CholeskyFactorizer_LeftLooking")+"<"+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // class CholeskyFactorizer_LeftLooking
        
    } // namespace Sparse
    
} // namespace Tensors
