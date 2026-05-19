#pragma once

namespace Tensors
{
    namespace Sparse
    {
        
        template<IntQ LInt>
        struct Position
        {
            const LInt index;
            const bool foundQ;
        };
        
        template<IntQ Int, IntQ LInt, Parallel_T parQ> class BinaryMatrixCSR;
        
        template<IntQ Int_, IntQ LInt_, Parallel_T parQ_/* = Parallel*/>
        class PatternCSR
        {
            // Int  - an integer type capable of storing both the number of rows and columns
            // LInt - a potentially longer integer type capable of storing the absolute number of nonzeros.
            
        public:
            
            using Int  = Int_;
            using LInt = LInt_;
            
            static constexpr Parallel_T parQ = parQ_;
            
        protected:
            
            // I have to make these mutable so that methods that depend on SortInner can be called also from const instances of the class.
            mutable Tensor1<LInt, Int> outer;
            mutable Tensor1< Int,LInt> inner;
            
            // I cannot make these const since swap would not work otherwise.
            Int m = 0;
            Int n = 0;
            
            Int thread_count = 1;
            
            
            // diag_ptr[i] is the first nonzero element in row i such that inner[diag_ptr[i]] >= i
            mutable Tensor1<LInt,Int> diag_ptr;
            mutable JobPointers<Int> job_ptr;
            mutable JobPointers<Int> upper_triangular_job_ptr;
            mutable JobPointers<Int> lower_triangular_job_ptr;
            
            
            mutable bool proven_inner_sortedQ         = false;
            mutable bool proven_duplicate_freeQ       = false;
            
            mutable bool diag_ptr_initialized = false;
            mutable bool job_ptr_initialized  = false;
            mutable bool upper_triangular_job_ptr_initialized  = false;
            mutable bool lower_triangular_job_ptr_initialized  = false;
            
            bool symmetric                    = false;
            
        public:
            
            friend class BinaryMatrixCSR<Int,LInt,parQ>;
            
            // TODO: Is this a good idea?
//            friend class BinaryMatrixCSR<Int,LInt,Parallel>;
//            friend class BinaryMatrixCSR<Int,LInt,Sequential>;
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            PatternCSR(
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer        { int_cast<Int>(m_+1),LInt(0)                  }
            ,   m            { int_cast<Int>(m_)                            }
            ,   n            { int_cast<Int>(n_)                            }
            ,   thread_count { TruncateThreadCount<Int,parQ>(thread_count_) }
            {
                outer[0] = LInt(0);
            }
            
            template<IntQ I_0, IntQ I_1, IntQ I_2, IntQ I_3>
            PatternCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   outer        { int_cast<Int>(m_+1), LInt(0)                 }
            ,   inner        { int_cast<LInt>(nnz_)                         }
            ,   m            { int_cast<Int>(m_)                            }
            ,   n            { int_cast<Int>(n_)                            }
            ,   thread_count { TruncateThreadCount<Int,parQ>(thread_count_) }
            {
                outer[0] = LInt(0);
            }
            
            template<IntQ J_0, IntQ J_1, IntQ I_0, IntQ I_1, IntQ I_3>
            PatternCSR(
                cptr<J_0> outer_,
                cptr<J_1> inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer        { m_+1                                         }
            ,   inner        { int_cast<LInt>(outer_[m_])                   }
            ,   m            { int_cast<Int>(m_)                            }
            ,   n            { int_cast<Int>(n_)                            }
            ,   thread_count { TruncateThreadCount<Int,parQ>(thread_count_) }
            {
                outer.Read(outer_);
                inner.Read(inner_);
            }
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            PatternCSR(
                cref<Tensor1<LInt, Int>> outer_,
                cref<Tensor1< Int,LInt>> inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer        { outer_                                       }
            ,   inner        { inner_                                       }
            ,   m            { int_cast<Int>(m_)                            }
            ,   n            { int_cast<Int>(n_)                            }
            ,   thread_count { TruncateThreadCount<Int,parQ>(thread_count_) }
            {}
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            PatternCSR(
                Tensor1<LInt, Int> && outer_,
                Tensor1< Int,LInt> && inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer        { std::move(outer_)                            }
            ,   inner        { std::move(inner_)                            }
            ,   m            { int_cast<Int>(m_)                            }
            ,   n            { int_cast<Int>(n_)                            }
            ,   thread_count { TruncateThreadCount<Int,parQ>(thread_count_) }
            {}
            
            // Default constructor
            PatternCSR() = default;
            // Destructor
            virtual ~PatternCSR() = default;
            // Copy constructor
            PatternCSR( const PatternCSR & other ) = default;
            // Copy assignment operator
            PatternCSR & operator=( const PatternCSR & other ) = default;
            // Move constructor
            PatternCSR( PatternCSR && other ) = default;
            // Move assignment operator
            PatternCSR & operator=( PatternCSR && other ) = default;
            
//            // Copy constructor
//            PatternCSR( const PatternCSR & other )
//            :   outer           ( other.outer           )
//            ,   inner           ( other.inner           )
//            ,   m               ( other.m               )
//            ,   n               ( other.n               )
//            ,   thread_count { parQ == Parallel ? int_cast<Int>(thread_count_) : Int(1) }
//            ,   proven_inner_sortedQ    ( other.proven_inner_sortedQ    )
//            ,   proven_duplicate_freeQ  ( other.proven_duplicate_freeQ  )
//            ,   symmetric       ( other.symmetric       )
//            {
//                logprint("Copy of " + ClassName() + " of size {" + ToString(other.m) + ", " + ToString(other.n) + "}, nnz = " + ToString(other.NonzeroCount()));
//            }
//
//            
//            // Copy assignment operator
//            PatternCSR & operator=( PatternCSR other ) // Passing by value is okay, because of copy elision.
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
//            PatternCSR( PatternCSR && other ) noexcept
//            : PatternCSR()
//            {
//                swap(*this, other);
//            }
            
            // We do not need a move-assignment operator, because we use the copy-swap idiom!
            
            // Swap function
            friend void swap (PatternCSR &A, PatternCSR &B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( A.outer,          B.outer          );
                swap( A.inner,          B.inner          );
                swap( A.m,              B.m              );
                swap( A.n,              B.n              );
                swap( A.thread_count,   B.thread_count   );
                swap( A.proven_inner_sortedQ,   B.proven_inner_sortedQ   );
                swap( A.proven_duplicate_freeQ, B.proven_duplicate_freeQ );
                swap( A.symmetric,      B.symmetric      );
            }
            
            template<IntQ ExtInt>
            PatternCSR(
                const ExtInt * const * const idx,
                const ExtInt * const * const jdx,
                const LInt   *         const entry_counts,
                const Int list_count,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            :   PatternCSR ( m_, n_, list_count )
            {
                
                FromPairs( idx, jdx, entry_counts, list_count, final_thread_count, compressQ, symmetrizeQ );
            }
            
            template<IntQ ExtInt>
            PatternCSR(
                const LInt nnz_,
                const ExtInt  * const i,
                const ExtInt  * const j,
                const Int m_,
                const Int n_,
                const Int thread_count_,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            :   PatternCSR ( m_, n_, thread_count_ )
            {
                Tensor1<const ExtInt *,Int> idx    (thread_count);
                Tensor1<const ExtInt *,Int> jdx    (thread_count);
                Tensor1<      LInt    ,Int> counts (thread_count);
                
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const LInt begin = JobPointer<LInt>(nnz_, thread_count, thread    );
                    const LInt end   = JobPointer<LInt>(nnz_, thread_count, thread + 1);
                    
                    idx[thread] = &i[begin];
                    jdx[thread] = &j[begin];
                    counts[thread] = end-begin;
                }
                
                FromPairs( idx.data(), jdx.data(), counts.data(), Int(1), TruncateThreadCount(thread_count), compressQ, symmetrizeQ );
            }
            
            template<IntQ ExtInt>
            PatternCSR(
                cref<std::vector<ExtInt>> idx,
                cref<std::vector<ExtInt>> jdx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            :   PatternCSR ( m_, n_, Int(1) )
            {
                const ExtInt * i = idx.data();
                const ExtInt * j = jdx.data();
                
                Tensor1<LInt,Int> entry_counts (1, int_cast<LInt>(idx.size()));

                FromPairs( &i, &j, entry_counts.data(), Int(1), final_thread_count, compressQ, symmetrizeQ );
            }
            
            template<IntQ ExtInt>
            PatternCSR(
                cref<std::vector<std::vector<ExtInt>>> idx,
                cref<std::vector<std::vector<ExtInt>>> jdx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            // TODO: TruncateThreadCount? Or does it break anything?
            :   PatternCSR ( m_, n_, int_cast<Int>(idx.size()) )
            {
                Int list_count = int_cast<Int>(idx.size());
                Tensor1<const ExtInt *,Int> i (list_count);
                Tensor1<const ExtInt *,Int> j (list_count);
                
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].data();
                    j[thread] = jdx[thread].data();
                    entry_counts[thread] = static_cast<LInt>(idx[thread].size());
                }
                
                FromPairs( i.data(), j.data(), entry_counts.data(), list_count, final_thread_count, compressQ, symmetrizeQ );
            }
            
            template<IntQ ExtInt>
            PatternCSR(
                cref<std::vector<PairAggregator<ExtInt,ExtInt,LInt>>> pairs,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            // TODO: TruncateThreadCount? Or does it break anything?
            :   PatternCSR ( m_, n_, static_cast<Int>(pairs.size()) )
            {
                Int list_count = static_cast<Int>(pairs.size());
                Tensor1<const ExtInt *,Int> i (list_count);
                Tensor1<const ExtInt *,Int> j (list_count);
                
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    const Size_T t = static_cast<Size_T>(thread);
                    i[thread] = pairs[t].data_0();
                    j[thread] = pairs[t].data_1();
                    
                    entry_counts[thread] = static_cast<LInt>(pairs[t].Size());
                }
                
                FromPairs(
                    i.data(), j.data(), entry_counts.data(),
                    list_count, final_thread_count, compressQ, symmetrizeQ
                );
            }
            
            template<IntQ ExtInt>
            PatternCSR(
                cref<PairAggregator<ExtInt,ExtInt,LInt>> pairs,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            :   PatternCSR ( m_, n_, int_cast<LInt>(1) )
            {
                LInt entry_counts = int_cast<LInt>(pairs.Size());

                const ExtInt * const i = pairs.data_0();
                const ExtInt * const j = pairs.data_1();
                
                FromPairs(
                    &i,&j,&entry_counts,
                    ExtInt(1),final_thread_count,compressQ,symmetrizeQ
                );
            }
            
            
        public:
            
            Int ThreadCount() const
            {
                return thread_count;
            }
            
            void SetThreadCount( const Int thread_count_ )
            {
                thread_count = TruncateThreadCount<Int,parQ>(thread_count_);
                
                job_ptr_initialized = false;
            }
            
            bool ProvenInnerSortedQ() const
            {
                return proven_inner_sortedQ;
            }
            
            bool ProvenDuplicateFreeQ() const
            {
                return proven_duplicate_freeQ;
            }
            
        protected:
            
            void RequireJobPtr() const
            {
                if( job_ptr_initialized ) { return; }
                
//                TOOLS_PTIMER(timer,MethodName("RequireJobPtr"));
                
                job_ptr = JobPointers<Int>( m, outer.data(), thread_count, false );
                
                job_ptr_initialized = true;
            }
            
            void CheckOrdering() const
            {
                if( !proven_inner_sortedQ )
                {
                    eprint(MethodName("CheckOrdering")+": Column indices might not be sorted appropriately. Better call SortInner() first.");
                }
            }
            
            void RequireDiag() const
            {
                if( diag_ptr_initialized ) { return; }
                
//                TOOLS_PTIMER(timer,ClassName()+"::RequireDiag");
                
                SortInner();
                
                if( outer.Last() <= 0 )
                {
                    diag_ptr = Tensor1<LInt,Int>( outer.data()+1, m );
                }
                else
                {
                    RequireJobPtr();
                    
                    diag_ptr = Tensor1<LInt,Int>( m );
                    
                    Do<parQ>(
                        [=,this]( const Int i )
                        {
                            const LInt k_begin = outer[i  ];
                            const LInt k_end   = outer[i+1];
                            
                            LInt k = k_begin;
                            
                            while( (k < k_end) && (inner[k] < i)  )
                            {
                                ++k;
                            }
                            diag_ptr[i] = k;
                        },
                        job_ptr
                    );
                }
                
                diag_ptr_initialized = true;
            }
            
            void RequireUpperTriangularJobPtr() const
            {
                if( (m <= Int(0)) || upper_triangular_job_ptr_initialized )
                {
                    return;
                }
                
//                TOOLS_PTIMER(timer,MethodName("RequireUpperTriangularJobPtr"));
                
                RequireDiag();
                
                Tensor1<LInt,Int> costs (m + Int(1));
                costs[0] = 0;
                
                Do<parQ>(
                    [this,&costs]( const Int i )
                    {
                        costs[i+Int(1)] = outer[i+Int(1)] - diag_ptr[i];
                    },
                    job_ptr
                );
                
                // TODO: Check what works better.
                costs.Accumulate( thread_count );
//                costs.Accumulate(); // I think I prefer the sequential code here.
                
                upper_triangular_job_ptr = JobPointers( m, costs.data(), thread_count, false );
                
                upper_triangular_job_ptr_initialized = true;
            }
            
            void RequireLowerTriangularJobPtr() const
            {
                if( (m <= Int(0)) || lower_triangular_job_ptr_initialized )
                {
                    return;
                }
                TOOLS_PTIMER(timer,ClassName()+"::RequireLowerTriangularJobPtr");
                
                RequireDiag();
                
                Tensor1<LInt,Int> costs (m + Int(1));
                costs[0] = 0;
                
                Do<parQ>(
                    [this,&costs]( const Int i )
                    {
                        costs[i + Int(1)] = diag_ptr[i] - outer[i];
                    },
                    job_ptr
                );
                
                // TODO: Check what works better.
                costs.template Accumulate<parQ>( thread_count );
//                costs.Accumulate(); // I think I prefer the sequential code here.
                
                lower_triangular_job_ptr = JobPointers( m, costs.data(), thread_count, false );
                
                lower_triangular_job_ptr_initialized = true;
            }
            
        public:
            
            Int RowCount() const
            {
                return m;
            }
            
            Int ColCount() const
            {
                return n;
            }
            
            LInt NonzeroCount() const
            {
                return inner.Size();
            }
            
            LInt NonzeroCount( const Int i ) const
            {
                return outer(i+Int(1)) - outer(i);
            }
            
            mref<Tensor1<LInt,Int>> Outer()
            {
                return outer;
            }
            
            cref<Tensor1<LInt,Int>> Outer() const
            {
                return outer;
            }
            

            mref<LInt> Outer( const Int i )
            {
#ifndef TENSORS_BOUND_CHECKS
                if( i < Int(0) || i >= outer.Size() )
                {
                    eprint(this->ClassName()+"::Outer(" + ToString(i) + "): Access out of bounds.");
                }
#endif
                return outer[i];
            }
            
            cref<LInt> Outer( const Int i ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                if( i < Int(0) || i >= outer.Size() )
                {
                    eprint(this->ClassName()+"::Outer(" + ToString(i) + "): Access out of bounds.");
                }
#endif
                return outer[i];
            }
            
            
            mref<Tensor1<Int,LInt>> Inner()
            {
                return inner;
            }
            
            cref<Tensor1<Int,LInt>> Inner() const
            {
                return inner;
            }

            mref<Int> Inner( const LInt k )
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < LInt(0) || k >= inner.Size() )
                {
                    eprint(this->ClassName()+"::Inner(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return inner[k];
            }
            
            cref<Int> Inner( const LInt k ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < LInt(0) || k >= inner.Size() )
                {
                    eprint(this->ClassName()+"::Inner(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return inner[k];
            }
            
            cref<Tensor1<LInt,Int>> Diag() const
            {
                RequireDiag();
                
                return diag_ptr;
            }
            
            
            cref<LInt> Diag( const Int i ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                if( i < Int(0) || i >= diag_ptr.Size() )
                {
                    eprint(this->ClassName()+"::Diag(" + ToString(i) + "): Access out of bounds.");
                }
#endif
                return diag_ptr[i];
            }
            
            cref<JobPointers<Int>> JobPtr() const
            {
                RequireJobPtr();
                
                return job_ptr;
            }
            
            
            
            cref<JobPointers<Int>> UpperTriangularJobPtr() const
            {
                RequireUpperTriangularJobPtr();
                
                return upper_triangular_job_ptr;
            }
            
            
            cref<JobPointers<Int>> LowerTriangularJobPtr() const
            {
                RequireLowerTriangularJobPtr();
                
                return lower_triangular_job_ptr;
            }
            
            
        protected:
            
            [[nodiscard]] Tensor2<LInt,Int> CreateTransposeCounters() const
            {
//                TOOLS_PTIMER(timer,ClassName()+"::CreateTransposeCounters");
                
                RequireJobPtr();
                
                Tensor2<LInt,Int> counters ( thread_count, n, LInt(0) );
                
                if( !WellFormedQ() )
                {
                    return Tensor2<LInt,Int>();
                }
                
                // Use counting sort to sort outer indices of output matrix.
                // https://en.wikipedia.org/wiki/Counting_sort
                
                Do<parQ>(
                    [&,this]( const Int thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        mptr<LInt> c = counters.data(thread);
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt jj_begin = outer[i  ];
                            const LInt jj_end   = outer[i+1];
                            
                            for( LInt jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const Int j = inner[jj];
                                ++c[j];
                            }
                        }
                    },
                    thread_count, thread_count
                );
                
                AccumulateAssemblyCounters<parQ>( counters );
                
                return counters;
            }
            
        public:
            
            virtual void SortInner() const
            {
                // Sorts the column indices of each matrix row.
                
                if( proven_inner_sortedQ ) { return; }
                
                TOOLS_PTIMER(timer,MethodName("SortInner"));
                
                if( !WellFormedQ() ) { return; }
                    
                Do<parQ>(
                    [this]( const Int i )
                    {
                        // This sort uses sorting nets for small lists. This increases compile time, but sorts very sparse matrices faster.
                        Sort( &inner[outer[i]], &inner[outer[i+1]], std::less<LInt>() );
                    },
                    JobPtr()
                );
                
                proven_inner_sortedQ = true;
            }
            
    public:

        virtual void Compress() const
        {
            // Removes duplicate {i,j}-pairs.
            
            TOOLS_PTIMER(timer,MethodName("Compress"));
            
            Tensor1< Int,LInt> values;  // a dummy
            Tensor1<LInt,LInt> C_outer; // a dummy
            
            this->template Compress_impl<false,false>( outer, inner, values, C_outer );
        }

            
//##########################################################################
//####          Matrix Multiplication
//##########################################################################
            
        protected:
            
            PatternCSR DotBinary_( cref<PatternCSR> B ) const
            {
                TOOLS_PTIMER(timer,MethodName("DotBinary_"));
                
                if( !WellFormedQ() || !B.WellFormedQ() )
                {
                    return PatternCSR ();
                }
                
                RequireJobPtr();
                
                // Create counters for counting sort
                
                Tensor2<LInt,Int> counters ( thread_count, m, LInt(0) );
                
                // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                // https://en.wikipedia.org/wiki/Counting_sort
                
                Do<parQ>(
                    [&,this]( const Int thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        mptr<LInt> c = counters.data(thread);
                        
                        cptr<LInt> A_outer = Outer().data();
                        cptr<Int>  A_inner = Inner().data();
                        cptr<LInt> B_outer = B.Outer().data();
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            LInt c_i = 0;
                            
                            const LInt jj_begin = A_outer[i  ];
                            const LInt jj_end   = A_outer[i+1];
                            
                            for( LInt jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const Int j = A_inner[jj];
                                
                                c_i += (B_outer[j+1] - B_outer[j]);
                            }
                            
                            c[i] = c_i;
                        }
                    },
                    thread_count, thread_count
                );
                
                AccumulateAssemblyCounters<parQ>(counters);
                
                const LInt nnz = counters[thread_count-1][m-1];
                
                PatternCSR C ( m, B.ColCount(), nnz, thread_count );
                
                copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                
                // Counting sort.
                
                Do<parQ>(
                    [&,this]( const Int thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        mptr<LInt> c        = counters.data(thread);
                        
                        cptr<LInt> A_outer  = Outer().data();
                        cptr< Int> A_inner  = Inner().data();
                        
                        cptr<LInt> B_outer  = B.Outer().data();
                        cptr< Int> B_inner  = B.Inner().data();
                        
                        mptr< Int> C_inner  = C.Inner().data();
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt jj_begin = A_outer[i  ];
                            const LInt jj_end   = A_outer[i+1];
                            
                            for( LInt jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const Int j = A_inner[jj];
                                
                                const LInt kk_begin = B_outer[j  ];
                                const LInt kk_end   = B_outer[j+1];
                                
                                for( LInt kk = kk_end; kk --> kk_begin; )
                                {
                                    const Int k = B_inner[kk];
                                    const LInt pos = --c[i];
                                    
                                    C_inner [pos] = k;
//                                        C_values[pos] = A_values[jj] * B_values[kk];
                                }
                            }
                        }
                    },
                    thread_count, thread_count
                );
                
                // Finished expansion phase (counting sort).
                
                // Finally we rwo-sort inner and remove duplicates in inner and values.
                C.Compress();
                
                return C;
            }
            
        protected:
            
            // Assume all nonzeros are equal to 1.
            template<Int NRHS = VarSize, typename alpha_T, typename X_T, typename beta_T, typename Y_T>
            void Dot_(
                const alpha_T alpha, cptr<X_T> X, const Int ldX,
                const beta_T  beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = Int(1)
            ) const
            {
                if( !WellFormedQ() )
                {
                    wprint(ClassName()+"::Dot_: Not WellFormedQ(). Doing nothing.");
                }
                                
                SparseBLAS<Scalar::Real<X_T>,Int,LInt> sblas;
                
                sblas.template Multiply_DenseMatrix<NRHS>(
                    outer.data(), inner.data(), nullptr,
                    m, n,
                    alpha, X, ldX,
                    beta,  Y, ldY,
                    nrhs, JobPtr()
                );
            }
            
            // Supply an external list of values.
            template<Int NRHS = VarSize, typename a_T, typename alpha_T, typename X_T, typename beta_T, typename Y_T>
            void Dot_(
                      cptr<a_T> values,
                      const alpha_T alpha, cptr<X_T> X, const Int ldX,
                      const beta_T  beta,  mptr<Y_T> Y, const Int ldY,
                      const Int   nrhs = Int(1)
            ) const
            {
                if( !WellFormedQ() )
                {
                    wprint(ClassName()+"::Dot_: Not WellFormedQ(). Doing nothing.");
                }
                
                SparseBLAS<a_T,Int,LInt> sblas;
                
                sblas.template Multiply_DenseMatrix<NRHS>(
                    outer.data(), inner.data(), values,
                    m, n,
                    alpha, X, ldX,
                    beta,  Y, ldY,
                    nrhs, JobPtr()
                );
            }
            
//##########################################################################
//####          Conversion Operations
//##########################################################################
            
        public:
            
            // Supply an external list of values.
            template<typename A_T, typename values_T>
            void WriteDenseWithValues(
                cptr<values_T> values, mptr<A_T> A, const Int ldA
            ) const
            {
                Do<parQ>(
                    [this,values,A,ldA]( const Int i )
                    {
                        const LInt k_begin = outer[i    ];
                        const LInt k_end   = outer[i + 1];
                        
                        zerofy_buffer( &A[i * ldA], n );
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            A[i * ldA + inner[k]] += values[k];
                        }
                    },
                    JobPtr()
                );
            }
            
            template<typename A_T>
            void WriteDense( mptr<A_T> A, const Int ldA ) const
            {
                Do<parQ>(
                    [this,A,ldA]( const Int i )
                    {
                        const LInt k_begin = outer[i    ];
                        const LInt k_end   = outer[i + 1];
                        
                        zerofy_buffer( &A[i * ldA], n );
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            A[i * ldA + inner[k]] += A_T(1);
                        }
                    },
                    JobPtr()
                );
            }
            
            // Supply an external list of values.
            template<typename values_T, typename A_T = values_T >
            Tensor2<A_T,LInt> ToTensor2WithValues( cptr<values_T> values ) const
            {
                Tensor2<A_T,LInt> A ( m, n );
                
                WriteDenseWithValues( values, A.data(), n );
                
                return A;
            }
            
            // Supply an external list of values.
            template<typename A_T = bool>
            Tensor2<A_T,LInt> ToTensor2() const
            {
                Tensor2<A_T,LInt> A ( m, n );
                
                WriteDense( A.data(), n );
                
                return A;
            }
            
            std::tuple<Tensor1<LInt,Int>,Tensor1<Int,LInt>,Int,Int> Disband()
            {
                Tensor1<LInt, Int> outer_  = std::move(outer);
                Tensor1< Int,LInt> inner_  = std::move(inner);
                Int m_ = m;
                Int n_ = n;
                
                *this = PatternCSR();
                
                return { outer_, inner_, m_, n_ };
            }
            
//##########################################################################
//####          Lookup Operations
//##########################################################################
            
            
        private:
            
            void BoundCheck( const Int i, const Int j ) const
            {
                if( std::cmp_less(i,Int(0)) || std::cmp_greater(i,m) )
                {
                    eprint(ClassName()+": Row index " + ToString(i) + " is out of bounds [ 0, " + ToString(m) +" [.");
                }
                if( std::cmp_less(j,Int(0)) || std::cmp_greater(j,n) )
                {
                    eprint(ClassName()+": Column index " + ToString(j) + " is out of bounds [ 0, " + ToString(n) +" [.");
                }
            }
            
        public:
            
            bool InnerSortedQ() const
            {
                return proven_inner_sortedQ;
            }
            
            void AssumeInnerSorted()
            {
                proven_inner_sortedQ = true;
            }
            
            void AssumeInnerUnsorted()
            {
                proven_inner_sortedQ = false;
            }
            
            bool NonzeroPositionQ( const Int i, const Int j ) const
            {
                return FindNonzeroPosition( i, j ).foundQ;
            }
            
            Sparse::Position<LInt> FindNonzeroPosition( const Int i, const Int j ) const
            {
                // Looks up the entry {i,j}. If existent, its index within the list of nonzeroes is returned. Otherwise, a negative number is returned (-1 if simply not found and -2 if i is out of bounds).
                
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i,j);
#endif
                
                constexpr LInt threshold = 6;
                
                if( (Int(0) <= i) && (i < m) )
                {
                    cptr<Int> A_inner = inner.data();
                    
                    LInt L = outer[i  ];
                    LInt R = outer[i+1];
                    
                    // We need to be careful here
                    // since outer[i+1] could be zero,
                    // and LInt could be an unsigned integer type.
                    
                    if( L == R )
                    {
                        // No matrix entries in this row. We can abort.
                        return Sparse::Position<LInt>{0, false};
                    }
                    
                    --R;
                    
                    if( proven_inner_sortedQ && ( L + threshold > R ) )
                    {
                        if( j < A_inner[L] )
                        {
                            return Sparse::Position<LInt>{0, false};
                        }
                        
                        if( j > A_inner[R] )
                        {
                            return Sparse::Position<LInt>{0, false};
                        }
                        
                        while( L < R )
                        {
                            const LInt k = R - (R-L)/Int(2);
                            const Int col = A_inner[k];
                            
                            if( col > j )
                            {
                                R = k-1;
                            }
                            else
                            {
                                L = k;
                            }
                        }
                    }
                    else
                    {
                        // If unsorted or if only few entries
                        // are around in this row, we just do a linear search.
                        while( (L < R) && (A_inner[L] < j) )
                        {
                            ++L;
                        }
                    }
                    
                    return (A_inner[L]==j) 
                        ? Sparse::Position<LInt>{L, true }
                        : Sparse::Position<LInt>{0, false};
                }
                else
                {
                    wprint(ClassName()+"::FindNonzeroPosition: Row index i = "+ToString(i)+" is out of bounds {0,"+ToString(m)+"}.");
                    return Sparse::Position<LInt>{0, false};
                }
            }
            
            
            template<typename S, typename T, typename J>
            void FillLowerTriangleFromUpperTriangle( mref<std::map<S,Tensor1<T,J>>> values )
            {
                TOOLS_PTIMER(timer,MethodName("FillLowerTriangleFromUpperTriangle"));
                
                if( !WellFormedQ() ) { return; }
                
                SortInner();
                
                cptr<LInt> A_diag  = Diag().data();
                cptr<LInt> A_outer = Outer().data();
                cptr<Int>  A_inner = Inner().data();
                
                Do<parQ>(
                    [=,this]( const Int i )
                    {
                        const LInt k_begin = A_outer[i];
                        const LInt k_end   = A_diag [i];
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            const Int j = A_inner[k];
                            
                            LInt L = A_diag [j];
                            LInt R = A_outer[j+1]-1;
                            
                            while( L < R )
                            {
                                const LInt M   = R - (R-L)/Int(2);
                                const  Int col = A_inner[M];
                                
                                if( col > i )
                                {
                                    R = M-1;
                                }
                                else
                                {
                                    L = M;
                                }
                            }
                            
                            for( auto & f : values )
                            {
                                f.second[k] = f.second[L];
                            }
                            
                        } // for( Int k = k_begin; k < k_end; ++k )
                    },
                    LowerTriangularJobPtr()
                );
            }
            
            bool WellFormedQ() const
            {
                bool wellformed = ( ( outer.Size() > Int(1) ) && ( outer.Last() > LInt(0) ) );
                
                if( !wellformed )
                {
                    wprint(ClassName()+"::WellFormedQ: Matrix is not well formed.");
                    
                    TOOLS_LOGDUMP(m);
                    TOOLS_LOGDUMP(n);
                    TOOLS_LOGDUMP(outer.Size());
                    TOOLS_LOGDUMP(inner.Size());
                    
                    if( outer.Size() > Int(0) )
                    {
                        TOOLS_LOGDUMP(outer.First());
                        TOOLS_LOGDUMP(outer.Last());
                    }
                }
                
                return wellformed;
            }
            
            LInt NonsymmetricCount() const
            {
                // This routine is not optimized.
                // Only meant for debugging.
                
                LInt unmatched_count = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    const LInt k_begin = outer[i    ];
                    const LInt k_end   = outer[i + 1];
                    
                    for( LInt k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];
                        
                        if( i < j )
                        {
                            const bool foundQ = NonzeroPositionQ(j,i);
                            
                            unmatched_count += (!foundQ);
                        }
                    }
                }
                
                return unmatched_count;
            }
            
            
            template<IntQ ExtInt>
            void WriteNonzeroPositions( mptr<ExtInt> pos )
            {
                if( !std::in_range<ExtInt>(RowCount()) )
                {
                    eprint(MethodName("WriteNonzeroPositions") + ": RowCount() = " + ToString(RowCount()) + " is too large to fit into target type " + TypeName<ExtInt> +". Doing nothing.");
                    return;
                }
                
                if( !std::in_range<ExtInt>(ColCount()) )
                {
                    eprint(MethodName("WriteNonzeroPositions") + ": ColCount() = " + ToString(RowCount()) + " is too large to fit into target type " + TypeName<ExtInt> +". Doing nothing.");
                    return;
                }
                
                RequireJobPtr();
                
                Do<parQ>(
                    [pos,this]( const Int i )
                    {
                        const LInt k_begin = outer[i    ];
                        const LInt k_end   = outer[i + 1];
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            const Int j = inner[k];
                            
                            pos[Int(2) * k + Int(0)] = i;
                            pos[Int(2) * k + Int(1)] = j;
                        }
                    },
                    job_ptr
                );
            }
            
            template<IntQ ExtInt>
            void WriteNonzeroPositions( mptr<ExtInt> idx, mptr<ExtInt> jdx )
            {
                if( !std::in_range<ExtInt>(RowCount()) )
                {
                    eprint(MethodName("WriteNonzeroPositions") + ": RowCount() = " + ToString(RowCount()) + " is too large to fit into target type " + TypeName<ExtInt> +". Doing nothing.");
                    return;
                }
                
                if( !std::in_range<ExtInt>(ColCount()) )
                {
                    eprint(MethodName("WriteNonzeroPositions") + ": ColCount() = " + ToString(RowCount()) + " is too large to fit into target type " + TypeName<ExtInt> +". Doing nothing.");
                    return;
                }
                
                RequireJobPtr();
                
                Do<parQ>(
                    [idx,jdx,this]( const Int i )
                    {
                        const LInt k_begin = outer[i    ];
                        const LInt k_end   = outer[i + 1];
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            const Int j = inner[k];
                            
                            idx[k] = i;
                            jdx[k] = j;
                        }
                    },
                    job_ptr
                );
            }
            
            Tiny::VectorList_AoS<2,Int,LInt> NonzeroPositions_AoS() const
            {
                Tiny::VectorList_AoS<2,Int,LInt> edges ( NonzeroCount() );

                RequireJobPtr();
                
                Do<parQ>(
                    [&edges,this]( const Int i )
                    {
                        const LInt k_begin = outer[i    ];
                        const LInt k_end   = outer[i + 1];
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            const Int j = inner[k];
                            
                            edges(k,0) = i;
                            edges(k,1) = j;
                        }
                    },
                    job_ptr
                );
                
                return edges;
            }
            
        public:
            
            static PatternCSR IdentityMatrix( const Int n, const Int thread_count = 1 )
            {
                PatternCSR A ( n, n, n, thread_count );
                A.Outer().iota();
                A.Inner().iota();
                
                return A;
            }
            
        public:
            
            virtual Int Dim( const bool dim )
            {
                return dim ? n : m;
            }
            
            
            virtual Int Dimension( const bool dim )
            {
                return Dim(dim);
            }
            
            std::string Stats() const
            {
                return std::string()
                + "\n==== "+ClassName()+" Stats ====\n\n"
                + " RowCount()      = " + ToString(RowCount()) + "\n"
                + " ColCount()      = " + ToString(ColCount()) + "\n"
                + " NonzeroCount()  = " + ToString(NonzeroCount()) + "\n"
                + " ThreadCount()   = " + ToString(ThreadCount()) + "\n"
                + " Outer().Size()  = " + ToString(Outer().Size()) + "\n"
                + " Inner().Size()  = " + ToString(Inner().Size()) + "\n"
                + "\n==== "+ClassName()+" Stats ====\n\n";
            }
            
#include "PatternCSR/Compress.hpp"
#include "PatternCSR/FromPairs.hpp"
            
        public:
            
            static std::string MethodName( const std::string & tag )
            {
                return ClassName() + "::" + tag;
            }
            
            static std::string ClassName()
            {
                return std::string("PatternCSR")
                + "<" + TypeName<Int>
                + "," + TypeName<LInt>
                + "," + ToString(parQ)
                + ">";
            }
            
        }; // PatternCSR
        
    } // namespace Sparse
    
} // namespace Tensors
