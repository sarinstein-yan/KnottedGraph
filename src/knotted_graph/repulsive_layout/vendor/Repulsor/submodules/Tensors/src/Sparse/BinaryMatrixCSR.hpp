#pragma once

namespace Tensors
{
    namespace Sparse
    {
        // TODO: Make permutation matrices more efficient: Add a flag if `outer = iota(m+1)`, avoid storing `outer` and adapt the indexing and `Dot` algorithms. Maybe.
        
        template<IntQ Int_, IntQ LInt_, Parallel_T parQ_ /*= Parallel*/>
        class BinaryMatrixCSR : public Sparse::PatternCSR<Int_,LInt_,parQ_>
        {
        private:
            
            using Base_T = Sparse::PatternCSR<Int_,LInt_,parQ_>;
            
        public:
            
            using Int  = Int_;
            using LInt = LInt_;
            
            static constexpr Parallel_T parQ = parQ_;
            
        protected:
            
            using Base_T::m;
            using Base_T::n;
            using Base_T::outer;
            using Base_T::inner;
            using Base_T::thread_count;
            using Base_T::job_ptr;
            using Base_T::upper_triangular_job_ptr;
            using Base_T::lower_triangular_job_ptr;
            using Base_T::Dot_;
            
        public:
            
            using Base_T::RowCount;
            using Base_T::ColCount;
            using Base_T::NonzeroCount;
            using Base_T::ThreadCount;
            using Base_T::SetThreadCount;
            using Base_T::Outer;
            using Base_T::Inner;
            using Base_T::JobPtr;
            using Base_T::Diag;
            using Base_T::SortInner;
            using Base_T::RequireDiag;
            using Base_T::RequireJobPtr;
            using Base_T::RequireUpperTriangularJobPtr;
            using Base_T::RequireLowerTriangularJobPtr;
            using Base_T::UpperTriangularJobPtr;
            using Base_T::LowerTriangularJobPtr;
            using Base_T::CreateTransposeCounters;
            using Base_T::WellFormedQ;
            using Base_T::ToTensor2;
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            BinaryMatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            {}
            
            template<IntQ I_0, IntQ I_1, IntQ I_2, IntQ I_3>
            BinaryMatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   Base_T( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<LInt>(nnz_), int_cast<Int>(thread_count_) )
            {}
            
            
            template<IntQ J_0, IntQ J_1, IntQ I_0, IntQ I_1, IntQ I_3>
            BinaryMatrixCSR(
                  const J_0 * const outer_,
                  const J_1 * const inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
            )
            :   Base_T( outer_, inner_, int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            {}
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            BinaryMatrixCSR(
                  const Tensor1<LInt, Int> & outer_,
                  const Tensor1< Int,LInt> & inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
            )
            :   Base_T( 
                    outer_,
                    inner_,
                    int_cast<Int>(m_),
                    int_cast<Int>(n_),
                    int_cast<Int>(thread_count_)
                )
            {}
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            BinaryMatrixCSR(
                  Tensor1<LInt, Int> && outer_,
                  Tensor1< Int,LInt> && inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
            )
            :   Base_T( 
                    std::move(outer_), 
                    std::move(inner_),
                    int_cast<Int>(m_),
                    int_cast<Int>(n_),
                    int_cast<Int>(thread_count_)
                )
            {}
            
            // Default constructor
            BinaryMatrixCSR() = default;
            // Destructor
            virtual ~BinaryMatrixCSR() override = default;
            // Copy constructor
            BinaryMatrixCSR( const BinaryMatrixCSR & other ) = default;
            // Copy assignment operator
            BinaryMatrixCSR & operator=( const BinaryMatrixCSR & other ) = default;
            // Move constructor
            BinaryMatrixCSR( BinaryMatrixCSR && other ) = default;
            // Move assignment operator
            BinaryMatrixCSR & operator=( BinaryMatrixCSR && other ) = default;
            
//            // Copy constructor
//            BinaryMatrixCSR( const BinaryMatrixCSR & other )
//            :   Base_T( other )
//            {
//                logprint("Copy of "+ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nnz = "+ToString(NonzeroCount()));
//            }
//            
//
//            
//            // (Copy-)assignment operator
//            BinaryMatrixCSR & operator=( BinaryMatrixCSR other ) // Pass by value is okay, because we use copy-swap idiom and copy elision.
//            {
//                // copy-and-swap idiom
//                // see https://stackoverflow.com/a/3279550/8248900 for details
//                
//                swap(*this, other);
//                
//                return *this;
//            }
//            
//            // Move constructor
//            BinaryMatrixCSR( BinaryMatrixCSR && other ) noexcept : BinaryMatrixCSR()
//            {
//                swap(*this, other);
//            }
//            
//            // We do not need a move-assignment operator, because we use the copy-swap idiom!
            
            // Swap function
            friend void swap( BinaryMatrixCSR & A, BinaryMatrixCSR & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( static_cast<Base_T&>(A), static_cast<Base_T&>(B) );
            }
            
            BinaryMatrixCSR(
                  const Int    * const * const idx,
                  const Int    * const * const jdx,
                  const LInt   *         const entry_counts,
                  const Int list_count,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T ( idx, jdx, entry_counts, list_count, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                const LInt nnz_,
                const Int  * const i,
                const Int  * const j,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            :   Base_T ( nnz_, i, j, m_, n_, thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  std::vector<Int> & idx,
                  std::vector<Int> & jdx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, jdx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  const std::vector<std::vector<Int>> & idx,
                  const std::vector<std::vector<Int>> & jdx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, jdx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  cref<std::vector<PairAggregator<Int, Int, LInt>>> idx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  cref<PairAggregator<Int, Int, LInt>> idx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            

            BinaryMatrixCSR Transpose() const
            {
                TOOLS_PTIMER(timer,MethodName("Transpose"));
                
                if( !this->WellFormedQ() ) { return BinaryMatrixCSR(); }
                
                Tensor2<LInt,Int> counters = CreateTransposeCounters();
                
                BinaryMatrixCSR<Int,LInt,parQ> B ( n, m, outer[m], thread_count );
                
                copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );

                Do<parQ>(
                    [&]( const Int thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        mptr<LInt> c = counters.data(thread);
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt k_begin = outer[i  ];
                            const LInt k_end   = outer[i+1];
                            
                            for( LInt k = k_end; k --> k_begin; )
                            {
                                const Int j = inner[k];
                                const LInt pos = --c[j];
                                B.Inner(pos) = i;
                            }
                        }
                    },
                    thread_count, thread_count
                );
                
                // Finished counting sort.
                
                // We only have to care about the correct ordering of inner indices and values.
                B.SortInner();
                
                return B;
            }
            
            std::tuple<Tensor1<LInt,Int>,Tensor1<Int,LInt>,Int,Int> Disband()
            {
                Tensor1<LInt,Int > outer_  = std::move(outer);
                Tensor1<Int ,LInt> inner_  = std::move(inner);
                Int m_ = m;
                Int n_ = n;
                
                *this = BinaryMatrixCSR();
                
                return { outer_, inner_, m_, n_ };
            }

//###########################################################
//####          Permute
//###########################################################
            
        public:
            
            // TODO: Bad design: non_parallelized permutations may spoil everything.
            template<Parallel_T p_parQ, Parallel_T q_parQ>
            Tensor1<LInt,LInt> Permute(
                const Permutation<Int,p_parQ> & p,     // row    permutation
                const Permutation<Int,q_parQ> & q      // column permutation
            )
            {
                TOOLS_PTIMER(timer,MethodName("Permute"));
                // Modifies `inner` and `outer` accordingly; returns the permutation to be applied to the nonzero values.
                
                if( (parQ == Parallel) && (p_parQ != Parallel) )
                {
                    wprint(MethodName("Permute") + ": Matrix is in parallel mode but permutation p is not.");
                }
                
                if( (parQ == Parallel) && (q_parQ != Parallel) )
                {
                    wprint(MethodName("Permute") + ": Matrix is in parallel mode but permutation q is not.");
                }
                
                this->proven_inner_sortedQ = true;
                this->diag_ptr_initialized = false;
                this->job_ptr_initialized  = false;
                this->upper_triangular_job_ptr_initialized = false;
                this->lower_triangular_job_ptr_initialized = false;
                
                Tensor1<LInt,LInt> perm = PermutePatternCSR( outer, inner, p, q, inner.Size(), true );
                
                return perm;
            }
            
//###########################################################
//####          Matrix Multiplication
//###########################################################
            
        public:
            
            
            [[nodiscard]] Sparse::BinaryMatrixCSR<Int,LInt,parQ> DotBinary( const BinaryMatrixCSR & B ) const
            {
                Sparse::BinaryMatrixCSR<Int,LInt,parQ> result;
                
                Base_T C = this->DotBinary_(B);
                
                swap(result,C);
                
                return result;
            }
            
            
            [[nodiscard]] Sparse::BinaryMatrixCSR<Int,LInt,parQ> Dot( const BinaryMatrixCSR & B ) const
            {
                return DotBinary(B);
            }

//###########################################################
//####          Matrix Multiplication
//###########################################################
            

            // Assume all nonzeros are equal to 1.
            template<Int NRHS = 0, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X, const Int ldX,
                const b_T beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Assume all nonzeros are equal to 1.
            template<Int NRHS = 0, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X,
                const b_T beta,  mptr<Y_T> Y,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            template<typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) != n || Y.Dim(0) != m )
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
                
                this->template Dot_<1>(
                    alpha, X.data(), Int(1),
                    beta,  Y.data(), Int(1),
                    Int(1)
                );
            }
            
            template<Int NRHS = 0, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor2<X_T,Int>> X,
                const b_T beta,  mref<Tensor2<Y_T,Int>> Y
            ) const
            {
                if( (X.Dim(0) != n) || (Y.Dim(0) != m) || (X.Dim(1) != Y.Dim(1)) )
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
                
                const Int nrhs = X.Dim(1);
                
                this->template Dot_<NRHS>( alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
            }
            
            
            
            // Supply an external list of values.
            template<Int NRHS = 0, typename T_ext, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                cptr<T_ext> ext_values,
                const a_T alpha, cptr<X_T> X, const Int ldX,
                const b_T beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            template<typename T_ext, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                cref<Tensor1<T_ext,LInt>> ext_values,
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( (X.Dim(0) != n) || (Y.Dim(0) != m) )
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
                
                this->template Dot_<1>( ext_values.data(),
                    alpha, X.data(), Int(1),
                    beta,  Y.data(), Int(1),
                    Int(1)
                );
            }
            
            template<Int NRHS = 0, typename T_ext, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                     cref<Tensor1<T_ext,LInt>> ext_values,
                     const a_T alpha, cref<Tensor2<X_T,Int>> X,
                     const b_T beta,  mref<Tensor2<Y_T,Int>> Y
                     ) const
            {
                if( (X.Dim(0) != n) || (Y.Dim(0) != m) || (X.Dim(1) != Y.Dim(1)) )
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
                
                const Int nrhs = X.Dim(1);
                this->template Dot_<NRHS>( ext_values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
            }
            
        public:
            
            static BinaryMatrixCSR IdentityMatrix(
                const Int n, const Int thread_count = 1
            )
            {
                BinaryMatrixCSR A ( n, n, n, thread_count );
                A.Outer().iota();
                A.Inner().iota();
                
                return A;
            }
            
            
        public:
            
            static std::string MethodName( const std::string & tag )
            {
                return ClassName() + "::" + tag;
            }
            
            static std::string ClassName()
            {
                return std::string("Sparse::BinaryMatrixCSR")
                + "<" + TypeName<Int>
                + "," + TypeName<LInt>
                + "," + ToString(parQ)
                + ">";
            }
            
        }; // BinaryMatrixCSR
        
    } // namespace Sparse
        
} // namespace Tensors
