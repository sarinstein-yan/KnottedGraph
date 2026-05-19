#pragma once

namespace Tensors
{
    namespace Sparse
    {
        
        template<typename Scal_, IntQ Int_, IntQ LInt_, Parallel_T parQ> class CholeskyDecomposition;
        
        template<typename Scal_, IntQ Int_, IntQ LInt_>
        class alignas(ObjectAlignment) CholeskyFactorizer_Multifrontal final
        {
            // Performs a left-looking factorization.
            
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
//            using Chol_T   = CholeskyDecomposition<Scal,Int,LInt>;
            
//            using Update_T = typename CholeskyDecomposition<Scal,Int,LInt,Sequential>::Update_T;
            
            using Update_T = Tensor2<Scal,LInt>;
            
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
            
            Scal             reg;    // Regularization parameter for the diagonal.
            
            cptr<Int>        child_ptr;
            cptr<Int>        child_idx;
            cptr<Int>        parents;
            
            cptr<Int>        SN_rp;
            cptr<LInt>       SN_outer;
            cptr<Int>        SN_inner;
            
            cptr<LInt>       SN_tri_ptr;
            mptr<Scal>       SN_tri_val;
            cptr<LInt>       SN_rec_ptr;
            mptr<Scal>       SN_rec_val;
            
            std::vector<Update_T> & SN_updates;
            
            // local data
            
            Tensor1<Int,Int>   idx;
            Tensor1<Int,Int>   lut;

            bool goodQ = true;
            
            // Monitors.
            
            float FetchFromA_time            = 0;
            float FetchFromChildren_time     = 0;
            float Factorize_time             = 0;
            float ComputeUpdateMatrix_time   = 0;
            
        public:
            
//            ~CholeskyFactorizer_Multifrontal() = default;
            
            ~CholeskyFactorizer_Multifrontal()
            {
//                TOOLS_DUMP(FetchFromA_time);
//                TOOLS_DUMP(FetchFromChildren_time);
//                TOOLS_DUMP(Factorize_time);
//                TOOLS_DUMP(ComputeUpdateMatrix_time);
            }
            
            template<Parallel_T parQ>
            CholeskyFactorizer_Multifrontal( CholeskyDecomposition<Scal,Int,LInt,parQ> & chol )
            // shared data
            :   n               ( chol.n                                        )
            ,   A_diag          ( chol.A.Diag().data()                          )
            ,   A_rp            ( chol.A.Outer().data()                         )
            ,   A_ci            ( chol.A.Inner().data()                         )
            ,   A_val           ( chol.A_val.data()                             )
            ,   reg             ( chol.reg                                      )
            ,   child_ptr       ( chol.AssemblyTree().ChildPointers().data()    )
            ,   child_idx       ( chol.AssemblyTree().ChildIndices().data()     )
            ,   parents         ( chol.AssemblyTree().Parents().data()          )
            ,   SN_rp           ( chol.SN_rp.data()                             )
            ,   SN_outer        ( chol.SN_outer.data()                          )
            ,   SN_inner        ( chol.SN_inner.data()                          )
            ,   SN_tri_ptr      ( chol.SN_tri_ptr.data()                        )
            ,   SN_tri_val      ( chol.SN_data.tri_val.data()                   )
            ,   SN_rec_ptr      ( chol.SN_rec_ptr.data()                        )
            ,   SN_rec_val      ( chol.SN_data.rec_val.data()                   )
            ,   SN_updates      ( chol.SN_updates                               )
//            // local data
            ,   idx             ( chol.n                                        )
            ,   lut             ( chol.max_n_1                                  )
            {}
            
        public:
                
            // Factorization routine.
            void operator()( const Int s )
            {
                const  Int i_begin = SN_rp[s  ];
                const  Int i_end   = SN_rp[s+1];
                
                const LInt j_begin = SN_outer[s  ];
                const LInt j_end   = SN_outer[s+1];
                
                const  Int n_0     = i_end - i_begin;
                const  Int n_1     = int_cast<Int>(j_end - j_begin);
                
                
                assert_positive(n_0);
                
                if( n_0 <= izero )
                {
                    eprint(ClassName()+"::operator(): n_0<=0");
                    TOOLS_DUMP(s);
                    TOOLS_DUMP(SN_rp[s  ]);
                    TOOLS_DUMP(SN_rp[s+1]);
                }
                
                /* The layout of the frontal matrix is this:
                 *
                 *    /             \
                 *    |  U_0   U_1  |
                 *    |             |
                 *    |   X     V   |
                 *    \             /
                 *
                 *
                 * where:
                 *
                 * U_0 is an upper triangular matrix of size n_0 x n_0.
                 * U_1 is a  rectangular      matrix of size n_0 x n_1.
                 * V   is a  square           matrix of size n_1 x n_1.
                 *
                 * U_0 and U_1 belong to the supernode s and will persist in the factor U.
                 *
                 * The _update matrix_ V is only temporary.
                 * The parent of s will deconstruct V has soon as it fetched its update.
                 */
                
                mptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[s]];
                mptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[s]];
                
                // TODO: Can we somehow avoid zeroing out explicitly?
                SN_updates[static_cast<Size_T>(s)] = Update_T( n_1, n_1, Scal(0) );
                mptr<Scal> V = SN_updates[static_cast<Size_T>(s)].data();
                
//                if( n_1 > Int(0) )
//                {
//                    SN_updates[s] = (Scal*)calloc( n_1 * n_1, sizeof(Scal) );
//                }
                
//                if( n_1 > Int(0) )
//                {
//                    AlignedAllocator<Int>::Alloc( SN_updates[s], n_1 * n_1 );
//                    zerofy_buffer( SN_updates[s], n_1 * n_1 );
//                }
//                
//                mptr<Scal> V  = SN_updates[s];
                    
//                Time FetchFromA_start = Clock::now();
                
                FetchFromA( i_begin, i_end, j_begin, j_end, U_0, U_1);
                
//                FetchFromA_time += Tools::Duration( FetchFromA_start, Clock::now() );
                

                // We have to fetch the updates of all children of s.

//                Time FetchFromChildren_start = Clock::now();

                // Indices of a child are always a subset of the indices of parent.
                // So we can use a lookup table idx to translate between local indices.
                
                // Preparing the index lookup table idx.
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    idx[i] = i - i_begin; // index within U_0.
                }
                
                for( LInt j = j_begin; j < j_end; ++j )
                {
                    const Int j_global = SN_inner[j];
                    
                    idx[j_global] = static_cast<Int>(j - j_begin);  // index within U_1.
                }
                
                const Int child_begin = child_ptr[s    ];
                const Int child_end   = child_ptr[s + 1];
                
                for( Int child = child_begin; child < child_end; ++child )
                {
                    const Int t = child_idx[child];
                    
                    FetchFromChild( t, n_0, n_1, i_end, U_0, U_1, V );
                    
                    // Deallocate child's update buffer.
                    SN_updates[static_cast<Size_T>(t)] = Update_T();
                    
//                    if( SN_updates[t]!= nullptr )
//                    {
//                        free(SN_updates[t]);
//                        SN_updates[t] = nullptr;
//                    }
                    
//                    AlignedAllocator<Int>::Free(SN_updates[t]);
                }
                
//                FetchFromChildren_time += Tools::Duration( FetchFromChildren_start, Clock::now() );
                
                
//                Time Factorize_start = Clock::now();
                
                FactorizeSupernode( n_0, n_1, U_0, U_1 );
                
//                Factorize_time += Tools::Duration( Factorize_start, Clock::now() );
                
                
//                Time ComputeUpdateMatrix_start = Clock::now();
                
                ComputeUpdateMatrix( n_0, n_1, U_1, V );
                
//                ComputeUpdateMatrix_time += Tools::Duration( ComputeUpdateMatrix_start, Clock::now() );
            }
            
        protected:
            
            void FetchFromA(
                const  Int i_begin, const  Int i_end,
                const LInt l_begin, const LInt l_end,
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
            }
            
            
            void FetchFromChild(
                const Int t, // the supernode from which to fetch (child)
                const Int n_0, const Int n_1, 
                const Int i_end,
                mptr<Scal> U_0, mptr<Scal> U_1, mptr<Scal> V
            )
            {
                const LInt l_begin = SN_outer[t    ];
                const LInt l_end   = SN_outer[t + 1];
                
                cptr<Int> t_idx = &SN_inner[l_begin];
                
                const Int m_1 = static_cast<Int>(l_end - l_begin);
                
                
                // Update matrix of child node.
                cptr<Scal> W = SN_updates[static_cast<Size_T>(t)].data();
//                cptr<Scal> W = SN_updates[t];
                
                /* We have to scatter-add W into the frontal matrix of s:
                 *
                 *    /             \
                 *    |  U_0   U_1  |
                 *    |             |
                 *    |   X     V   |
                 *    \             /
                 */
                
                // Find the first column of the rectangular block of supernode t
                // that is mapped to reactangular block of supernode s.

                Int mid = 0;
                
                while( (mid < m_1) && (t_idx[mid] < i_end) )
                {
                    ++mid;
                }
                
                // TODO: It might be a good idea to avoid the large lookup table idx altogether.
                // TODO: Instead we cound just scan s_idx = &SN_inner[i_begin] = &SN_inner[i_end-n_0].
                // Remove one layer of indirection.
                // In particular, the hot part of lut will quite certainly fit into cache.
                for( Int i = 0; i < m_1; ++i )
                {
                    lut[i] = idx[t_idx[i]];
                }
                
//                for( Int i = 0; i < mid; ++i )
//                {
//                    const Int lut_i = lut[i];
//                    
//                    cptr<Scal> W_i = &W[ m_1 * i ];
//                    
//                    mptr<Scal> U_0_lut_i = &U_0[ n_0 * lut_i ];
//                    
//                    for( Int j = i; j < mid; ++j )
//                    {
//                        U_0_lut_i[ lut[j] ] += W_i[ j ];
//                    }
//                    
//                    mptr<Scal> U_1_lut_i = &U_1[ n_1 * lut_i ];
//                    
//                    for( Int j = mid; j < m_1; ++j )
//                    {
//                        U_1_lut_i[ lut[j] ] += W_i[ j ];
//                    }
//                }

                // Update of U_0;
                for( Int i = 0; i < mid; ++i )
                {
                    const Int lut_i = lut[i];
                    
                    cptr<Scal> W_i = &W[ m_1 * i ];
                    
                    mptr<Scal> U_0_lut_i = &U_0[ n_0 * lut_i ];
                    
                    for( Int j = i; j < mid; ++j )
                    {
                        U_0_lut_i[ lut[j] ] += W_i[ j ];
                    }
                }
                
                // Update of U_1;
                for( Int i = 0; i < mid; ++i )
                {
                    const Int lut_i = lut[i];
                    
                    cptr<Scal> W_i = &W[ m_1 * i ];
                    
                    mptr<Scal> U_1_lut_i = &U_1[ n_1 * lut_i ];
                    
                    for( Int j = mid; j < m_1; ++j )
                    {
                        U_1_lut_i[ lut[j] ] += W_i[ j ];
                    }
                }
                
                // Update of V;
                for( Int i = mid; i < m_1; ++i )
                {
                    const Int lut_i = lut[i];
                    
                    cptr<Scal> W_i = &W[ m_1 * i ];
                    
                    mptr<Scal> V_lut_i = &V[ n_1 * lut_i ];
                    
                    for( Int j = i; j < m_1; ++j )
                    {
                        V_lut_i[ lut[j] ] += W_i[ j ];
                    }
                }
            }
            
            void FactorizeSupernode( 
                const Int n_0, const Int n_1, mptr<Scal> U_0, mptr<Scal> U_1
            )
            {
                if( n_0 > ione )
                {
                    // Cholesky factorization of U_0
                    const int info = LAPACK::potrf<Layout::RowMajor,UpLo::Upper,false>(
                        n_0, U_0, n_0
                    );
                    
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
                    U_0[0] = Sqrt(Abs( U_0[0] ));
                    
                    scale_buffer( Inv<Scal>( U_0[0] ), U_1, n_1 );
                }
            }
            
            
            void ComputeUpdateMatrix(
                const Int n_0, const Int n_1, mptr<Scal> U_1, mptr<Scal> V
            )
            {
                if( (n_0 > Int(0)) && (n_1 > Int(0)) )
                {
                    if( n_0 > ione )
                    {
                        // V -= upper(U_1^H * U_1)
                        BLAS::herk<Layout::RowMajor,UpLo::Upper,Op::ConjTrans>(
                            n_1, n_0,
                            -Scalar::Real<Scal>(1), U_1, n_1,
                             Scalar::Real<Scal>(1), V,   n_1
                        );
                    }
                    else
                    {
                        // V -= upper(U_1^H * U_1)
                        for( Int i = 0; i < n_1; ++i )
                        {
                            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus>(
                                -Conj(U_1[i]), U_1, one, &V[n_1 * i], n_1
                            );
                        }
                    }
                }
            }
            
            
        public:
            
            bool GoodQ() const
            {
                return goodQ;
            }
            
            std::string ClassName() const
            {
                return std::string("Sparse::CholeskyFactorizer_Multifrontal")+"<"+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // class CholeskyFactorizer_Multifrontal
        
    } // namespace Sparse
    
} // namespace Tensors
