//###########################################################
//####          IO routines
//###########################################################

public:

    template<Size_T NRHS = VarSize, typename B_T>
    void ReadRightHandSide( 
        cptr<B_T> B, const Int ldB,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        nrhs = Max( ione, nrhs_ );
        
        const std::string tag = ClassName()+"::ReadRightHandSide<" + TypeName<B_T> + ">(" + ToString(nrhs)+ ")";
        
        TOOLS_PTIMER(timer,tag);
        
        LInt full_size = static_cast<LInt>(n) * static_cast<LInt>(nrhs);
        
        if( X.Size() < full_size )
        {
            X         = VectorContainer_T(full_size);
            X_scratch = VectorContainer_T(
                static_cast<LInt>(max_n_1)*static_cast<LInt>(nrhs)
            );
        }

//        perm.Permute( B, ldB, X.data(), nrhs, Inverse::False, nrhs );
        
        perm.template PermuteCombine<NRHS>(
            Scalar::One <Scal>, B,        ldB,
            Scalar::Zero<Scal>, X.data(), nrhs,
            Inverse::False, nrhs
        );
    }

    template<typename X_T>
    void WriteSolution( mptr<X_T> X_, const Int ldX )
    {
        const std::string tag = ClassName()+"::WriteSolution<" + TypeName<X_T> + "> (" + ToString(nrhs)+ ")";
        
        TOOLS_PTIMER(timer,tag);
        
        perm.Permute( X.data(), nrhs, X_, ldX, Inverse::True, nrhs );
    }

    template<Size_T NRHS = VarSize, typename a_T, typename b_T, typename Y_T>
    void WriteSolution(
        const a_T alpha,
        const b_T beta,  mptr<Y_T> Y_, const Int ldY )
    {
        const std::string tag = ClassName()+"::WriteSolution"
            + "<" + ToString(NRHS)
            + "," + TypeName<a_T>
            + "," + TypeName<b_T>
            + "," + TypeName<Y_T>
            + ">(" + ToString(nrhs)+ ")";
        
        TOOLS_PTIMER(timer,tag);
        
        if ( nrhs == ione )
        {
            perm.template PermuteCombine<1>(
                alpha, X.data(), nrhs,
                beta,  Y_,       ldY,
                Inverse::True, nrhs
            );
        }
        else
        {
            perm.template PermuteCombine<NRHS>(
                alpha, X.data(), nrhs,
                beta,  Y_,       ldY,
                Inverse::True, nrhs
            );
        }
    }


//###########################################################
//####          Get routines
//###########################################################

    Int ThreadCount() const
    {
        return thread_count;
    }

    Int RowCount() const
    {
        return n;
    }

    Int ColCount() const
    {
        return n;
    }

    LInt NonzeroCount() const
    {
        return A.NonzeroCount();
    }

    Int RightHandSideCount() const
    {
        return nrhs;
    }

    //            const Matrix_T & GetL() const
    //            {
    //                return L;
    //            }


    cref<Matrix_T> GetU() const
    {
        std::string tag ( "U" );
        
        if( !InCacheQ(tag) )
        {
            // TODO: Debug this.
            
            Tensor1<LInt,Int> U_rp (n+1);
            U_rp[0] = 0;
            
            for( Int k = 0; k < SN_count; ++k )
            {
                const Int i_begin  = SN_rp[k  ];
                const Int i_end    = SN_rp[k+1];
                
                const LInt l_begin = SN_outer[k  ];
                const LInt l_end   = SN_outer[k+1];
                
    //                    const Int n_0 = i_end - i_begin;
                const Int n_1 = int_cast<Int>(l_end - l_begin);
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    U_rp[i+1] = U_rp[i] + (i_end-i) + n_1;
                }
            }
            
            Tensor1< Int,LInt> U_ci  (U_rp.Last(),0);
            Tensor1<Scal,LInt> U_val (U_rp.Last(),0);
            
            JobPointers<LInt> job_ptr ( SN_count, U_rp.data(), thread_count, false );
            
            Do<parQ>(
                [&,this]( const Int s )
                {
                    const Int i_begin  = SN_rp[s  ];
                    const Int i_end    = SN_rp[s+1];
                    
                    const LInt l_begin = SN_outer[s  ];
                    const LInt l_end   = SN_outer[s+1];
                    
                    const Int n_0 = i_end - i_begin;
                    const Int n_1 = int_cast<Int>(l_end - l_begin);

                    const Int start = U_rp[i_begin];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int delta = i-i_begin;

                        U_ci[start + delta] = i;
                    }
                    
                    
                    cptr<Scal> U_0 = &SN_data.tri_val[SN_tri_ptr[s]];
                    cptr<Scal> U_1 = &SN_data.rec_val[SN_rec_ptr[s]];
                    
                    // Copy first row of supernode.
                    copy_buffer( &SN_inner[l_begin], &U_ci[start+n_0], n_1 );
                    
                    // Copy triangular part.
                    copy_buffer( U_0, &U_val[U_rp[i_begin]      ], n_0 );
                    // Copy rectangular part.
                    copy_buffer( U_1, &U_val[U_rp[i_begin] + n_0], n_1 );
                    
                    // Copy the remaining rows of supernode.
                    for( Int i = i_begin+1; i < i_end; ++i )
                    {
                        // We are in the `i-loc`-th row of supernode s.
                        const Int i_loc = i-i_begin;
                        // Row `i_loc` of `U_0` has this many nonzero entries.
                        const Int n_0_i = i_end-i;
                        
                        copy_buffer( &U_ci[start+i_loc], &U_ci[U_rp[i]], n_0_i + n_1 );
                        
                        // Copy triangular part.
                        copy_buffer(
                            &U_0[n_0 * i_loc + i_loc],
                            &U_val[U_rp[i]],
                            n_0_i
                        );
                        
                        copy_buffer(
                            &U_1[n_1 * i_loc],
                            &U_val[U_rp[i] + n_0_i],
                            n_1
                        );
                    }
                },
                job_ptr
            );
            
            this->SetCache( tag,
                std::make_any<Matrix_T>(
                    std::move(U_rp), std::move(U_ci), std::move(U_val), n, n, thread_count
                )
            );
        }
        
        return this->template GetCache<Matrix_T>(tag);
    }

    cref<Matrix_T> GetFactor() const
    {
        return GetU();
    }

    cref<JobPointers<Int>> UJobPointers() const
    {
        std::string tag ( "UJobPointers" );
        
        if( !InPersistentCacheQ(tag) )
        {
            this->SetPersistentCache( tag,
                std::make_any<JobPointers<Int>>( SN_count, SN_rp.data(), thread_count, false )
            );
        }
        
        return this->template GetPersistentCache<JobPointers<Int>>(tag);
    }

    void WriteFactorDiagonal( mptr<Real> diag ) const
    {
        Do<parQ>(
            [&,this,diag]( const Int s )
            {
                const Int i_begin  = SN_rp[s  ];
                const Int i_end    = SN_rp[s+1];
                
                const Int n_0 = i_end - i_begin;

                cptr<Scal> U_0 = &SN_data.tri_val[SN_tri_ptr[s]];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    diag[i] = Re( U_0[(n_0+1) * (i-i_begin)] );
                }
            },
            UJobPointers()
        );
    }

    Real FactorLogDeterminant() const
    {
        const auto & job_ptr = UJobPointers();
        
        return ParallelDoReduce(
            [&job_ptr,this]( const Int thread )
            {
                Real log_det_local = 0;
                
                const Int s_begin  = job_ptr[thread  ];
                const Int s_end    = job_ptr[thread+1];
                
                for( Int s = s_begin; s < s_end; ++s )
                {
                    const Int i_begin  = SN_rp[s  ];
                    const Int i_end    = SN_rp[s+1];

                    const Int n_0 = i_end - i_begin;

                    cptr<Scal> U_0 = &SN_data.tri_val[SN_tri_ptr[s]];

                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        log_det_local += std::log( Re( U_0[(n_0+1) * (i-i_begin)] ) );
                    }
                }
                
                return log_det_local;
            },
            []( Real v, Real & r ) { r += v; },
            Scalar::Zero<Real>,
            thread_count
        );
    }

    Real LogDeterminant() const
    {
        return Scalar::Two<Real> * FactorLogDeterminant();
    }

    cref<BinaryMatrix_T> GetA() const
    {
        return A;
    }

    cref<Permutation_T> GetPermutation() const
    {
        return perm;
    }

    cref<Tensor1<LInt,LInt>> GetValuePermutation() const
    {
        return A_inner_perm;
    }

    Int SN_Count() const
    {
        return SN_count;
    }

    cref<Tensor1<Int,Int>> SN_RowPointers() const
    {
        return SN_rp;
    }

    cref<Tensor1<LInt,Int>> SN_Outer() const
    {
        return SN_outer;
    }

    cref<Tensor1<Int,LInt>> SN_Inner() const
    {
        return SN_inner;
    }

    cref<Tensor1<LInt,Int>> SN_TrianglePointers() const
    {
        return SN_tri_ptr;
    }

    cref<Tensor1<Scal,LInt>> SN_TriangleValues() const
    {
        return SN_data.tri_val;
    }

    cref<Tensor1<LInt,Int>> SN_RectanglePointers() const
    {
        return SN_rec_ptr;
    }

    cref<Tensor1<Scal,LInt>> SN_RectangleValues() const
    {
        return SN_data.rec_val;
    }

//    cref<std::vector<Update_T>> SN_UpdateValues() const
//    {
//        return SN_updates;
//    }

    cref<Tensor1<Int,Int>> RowToSN() const
    {
        return row_to_SN;
    }

    cref<Tensor1<Scal,LInt>> Values() const
    {
        return A_val;
    }

    void SetAmalgamationThreshold( const Int amalgamation_threshold_ )
    {
        if( amalgamation_threshold_ != amalgamation_threshold )
        {
            amalgamation_threshold = amalgamation_threshold_;
            SN_initializedQ        = false;
            SN_data.factorizedQ    = false;
            SN_data.goodQ          = false;
        }
    }

    Int AmalgamationThreshold() const
    {
        return amalgamation_threshold;
    }

    void SetSupernodeStrategy( cref<SupernodeStrategy_T> strategy )
    {
        if( strategy != SN_strategy )
        {
            SN_strategy         = strategy;
            SN_initializedQ     = false;
            SN_data.factorizedQ = false;
            SN_data.goodQ       = false;
        }
    }

    SupernodeStrategy_T GetSupernodeStrategy() const
    {
        return SN_strategy;
    }

    void SetFactorizationMethod( cref<FactorizationMethod_T> method )
    {
        factorization_method = method;
    }

    FactorizationMethod_T GetFactorizationMethod() const
    {
        return factorization_method;
    }
