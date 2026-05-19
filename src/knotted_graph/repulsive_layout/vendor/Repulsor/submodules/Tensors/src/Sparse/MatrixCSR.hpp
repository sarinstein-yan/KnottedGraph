#pragma once

#include <charconv>

namespace Tensors
{
    namespace Sparse
    {
        template<typename Scal_, IntQ Int_, IntQ LInt_, Parallel_T parQ_/* = Parallel*/>
        class MatrixCSR : public Sparse::PatternCSR<Int_,LInt_,parQ_>
        {
        private:
            
            using Base_T = Sparse::PatternCSR<Int_,LInt_,parQ_>;
            
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            using LInt = LInt_;
            
            static constexpr Parallel_T parQ = parQ_;
            
            using BinaryMatrix_T = Sparse::BinaryMatrixCSR<LInt,LInt,parQ>;
            using Assembler_T    = Sparse::BinaryMatrixCSR<LInt,LInt,parQ>;
        
        protected:
            
            using Base_T::m;
            using Base_T::n;
            using Base_T::outer;
            using Base_T::inner;
            using Base_T::thread_count;
            using Base_T::job_ptr;
            using Base_T::job_ptr_initialized;
            using Base_T::diag_ptr;
            using Base_T::proven_inner_sortedQ;
            using Base_T::proven_duplicate_freeQ;
            using Base_T::upper_triangular_job_ptr;
            using Base_T::lower_triangular_job_ptr;
            
            // I have to make this mutable so that SortInner and Compress can have the const attribute.
            mutable Tensor1<Scal,LInt> values;
            
            Assembler_T assembler;
            
        public:
            
            using Base_T::RowCount;
            using Base_T::ColCount;
            using Base_T::NonzeroCount;
            using Base_T::ThreadCount;
            using Base_T::SetThreadCount;
            using Base_T::Outer;
            using Base_T::Inner;
            using Base_T::RequireDiag;
            using Base_T::JobPtr;
            using Base_T::RequireJobPtr;
            using Base_T::RequireUpperTriangularJobPtr;
            using Base_T::RequireLowerTriangularJobPtr;
            using Base_T::UpperTriangularJobPtr;
            using Base_T::LowerTriangularJobPtr;
            using Base_T::CreateTransposeCounters;
            using Base_T::WellFormedQ;
            using Base_T::Dot_;
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            MatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            {}
            
            template<IntQ I_0, IntQ I_1, IntQ I_2, IntQ I_3>
            MatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   Base_T   ( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<LInt>(nnz_), int_cast<Int>(thread_count_) )
            ,   values ( int_cast<LInt>(nnz_) )
            {}
            
            template<typename S, IntQ J_0, IntQ J_1, IntQ I_0, IntQ I_1, IntQ I_3>
            MatrixCSR(
                const J_0 * const outer_,
                const J_1 * const inner_,
                const S   * const values_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T  ( outer_,  inner_, int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            ,   values  ( values_, outer_[int_cast<Int>(m_)] )
            {
                static_assert(ArithmeticQ<S>,"");
            }
            
            // CAUTION: This reserves memory for the nonzero values, but it does not initialize the nonzero values!
            template<IntQ J_0, IntQ J_1, IntQ I_0, IntQ I_1, IntQ I_3>
            MatrixCSR(
                const J_0 * const outer_,
                const J_1 * const inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T  ( outer_,  inner_, int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            ,   values  ( outer_[int_cast<Int>(m_)] )
            {}
            
            template<IntQ I_0, IntQ I_1, IntQ I_3>
            MatrixCSR(
                Tensor1<LInt, Int> && outer_,
                Tensor1< Int,LInt> && inner_,
                Tensor1<Scal,LInt> && values_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T ( std::move(outer_), std::move(inner_), int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            ,   values ( std::move(values_) )
            {}

            // Default constructor
            MatrixCSR() = default;
            // Destructor
            virtual ~MatrixCSR() override = default;
            // Copy constructor
            MatrixCSR( const MatrixCSR & other ) = default;
            // Copy assignment operator
            MatrixCSR & operator=( const MatrixCSR & other ) = default;
            // Move constructor
            MatrixCSR( MatrixCSR && other ) = default;
            // Move assignment operator
            MatrixCSR & operator=( MatrixCSR && other ) = default;

//            // Copy constructor
//            MatrixCSR( const MatrixCSR & other ) noexcept
//            :   Base_T    ( other           )
//            ,   values    ( other.values    )
//            ,   assembler ( other.assembler )
//            {
//                logprint("Copy of "+ ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nnz = "+ToString(NonzeroCount()));
//            }
//
//            
//            // (Copy-)assignment operator
//            MatrixCSR & operator=( MatrixCSR other ) noexcept // Pass by value is okay, because we use copy-swap idiom and copy elision.
//            {
//                // see https://stackoverflow.com/a/3279550/8248900 for details
//                
//                swap(*this, other);
//                
//                return *this;
//            }
//            
//            // Move constructor
//            MatrixCSR( MatrixCSR && other ) noexcept
//            :   MatrixCSR()
//            {
//                swap(*this, other);
//            }
            
            // We do not need a move-assignment operator, because we use the copy-swap idiom!
            
            
            friend void swap (MatrixCSR & A, MatrixCSR & B ) noexcept
            {
                using std::swap;
                
                swap( static_cast<Base_T&>(A), static_cast<Base_T&>(B) );
                swap( A.values,                B.values                );
                swap( A.assembler,             B.assembler             );
            }
            
            template<typename ExtScal, IntQ ExtInt>
            MatrixCSR(
                  const ExtInt  * const * const idx,
                  const ExtInt  * const * const jdx,
                  const ExtScal * const * const val,
                  const LInt    * const entry_counts,
                  const Int list_count,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
            )
            :   Base_T ( m_, n_, list_count )
            {
                FromTriples(
                    idx, jdx, val,
                    entry_counts, list_count, final_thread_count,
                    compressQ, symmetrizeQ, false
                );
            }
            
            
            template<IntQ ExtInt, typename ExtScal>
            MatrixCSR(
                const LInt nnz_,
                const ExtInt  * const i,
                const ExtInt  * const j,
                const ExtScal * const a,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ   = true,
                const int  symmetrizeQ = 0,
                const bool assemblerQ = false
            )
            :   Base_T ( m_, n_, thread_count )
            {
                Tensor1<const ExtInt  *,Int> idx    (thread_count);
                Tensor1<const ExtInt  *,Int> jdx    (thread_count);
                Tensor1<const ExtScal *,Int> val    (thread_count);
                Tensor1<      LInt  ,Int> counts (thread_count);
                
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const LInt begin = JobPointer<LInt>(nnz_,thread_count,thread);
                    const LInt end   = JobPointer<LInt>(nnz_,thread_count,thread + 1);
                    
                    idx[thread] = &i[begin];
                    jdx[thread] = &j[begin];
                    val[thread] = &a[begin];
                    counts[thread] = end-begin;
                }
                
                FromTriples( idx.data(), jdx.data(), val.data(), counts.data(), thread_count, thread_count, compressQ, symmetrizeQ, assemblerQ );
            }
            
            template<IntQ ExtInt, typename ExtScal>
            MatrixCSR(
                  cref<std::vector<std::vector<ExtInt>>>  idx,
                  cref<std::vector<std::vector<ExtInt>>>  jdx,
                  cref<std::vector<std::vector<ExtScal>>> val,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrizeQ = 0
            )
            :   Base_T ( m_, n_, int_cast<Int>(idx.size()) )
            {
                Int list_count = int_cast<Int>(idx.size());
                Tensor1<const ExtInt  *,Int> i      (list_count);
                Tensor1<const ExtInt  *,Int> j      (list_count);
                Tensor1<const ExtScal *,Int> a      (list_count);
                Tensor1<LInt,Int> entry_counts  (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].data();
                    j[thread] = jdx[thread].data();
                    a[thread] = val[thread].data();
                    entry_counts[thread] = int_cast<LInt>(idx[thread].size());
                }
                
                FromTriples(
                    i.data(), j.data(), a.data(),
                    entry_counts.data(), list_count,
                    final_thread_count, compressQ, symmetrizeQ, false
                );
            }
            
            template<IntQ ExtInt, typename ExtScal>
            MatrixCSR(
                  cref<std::vector<TripleAggregator<ExtInt,ExtInt,ExtScal,LInt>>> triples,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrizeQ = 0
            )
            :   Base_T ( m_, n_, int_cast<Int>(triples.size()) )
            {
                Int list_count = int_cast<Int>(triples.size());
                
                Tensor1<const ExtInt  *,Int> i (list_count);
                Tensor1<const ExtInt  *,Int> j (list_count);
                Tensor1<const ExtScal *,Int> a (list_count);
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    const Size_T t = static_cast<Size_T>(thread);
                    i[thread] = triples[t].data_0();
                    j[thread] = triples[t].data_1();
                    a[thread] = triples[t].data_2();
                    entry_counts[thread] = int_cast<LInt>(triples[t].Size());
                }
                
                FromTriples(
                    i.data(), j.data(), a.data(),
                    entry_counts.data(), list_count, final_thread_count,
                    compressQ, symmetrizeQ, false
                );
            }
            
            template<IntQ ExtInt, typename ExtScal>
            MatrixCSR(
                cref<TripleAggregator<ExtInt,ExtInt,ExtScal,LInt>> triples,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrizeQ = 0,
                const bool assemblerQ  = false
            )
            :   Base_T ( m_, n_, Int(1) )
            {
                LInt entry_counts = int_cast<LInt>(triples.Size());
                
                const ExtInt  * const i = triples.data_0();
                const ExtInt  * const j = triples.data_1();
                const ExtScal * const a = triples.data_2();

                FromTriples(
                    &i,&j,&a,&entry_counts,
                    Int(1), final_thread_count,
                    compressQ, symmetrizeQ, assemblerQ
                );
            }
            
        public:
            
            mref<Tensor1<Scal,LInt>> Values()
            {
                return values;
            }
            
            cref<Tensor1<Scal,LInt>> Values() const
            {
                return values;
            }
            
            mref<Tensor1<Scal,LInt>> Value()
            {
                return values;
            }
            
            cref<Tensor1<Scal,LInt>> Value() const
            {
                return values;
            }
            
            mref<Scal> Value( const LInt k )
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < LInt(0) || k >= values.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return values[k];
            }
            
            cref<Scal> Value( const LInt k ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < LInt(0) || k >= values.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return values[k];
            }
            
            
            Scal operator()( const Int i, const Int j ) const
            {
                const Sparse::Position<LInt> pos = this->FindNonzeroPosition(i,j);
                
                return ( pos.foundQ ) ? values[pos.index] : static_cast<Scal>(0);
            }
            
            cref<Assembler_T> Assembler() const
            {
                return assembler;
            }
            
        public:
            
            template<Tools::Op op,
                typename Return_T =
                std::conditional_t<
                    (op == Tools::Op::Re) || (op == Tools::Op::ReTrans)
                    ||
                    (op == Tools::Op::Im) || (op == Tools::Op::ImTrans)
                    ,
                    Scalar::Real<Scal>,
                    Scal
                >
            >
            MatrixCSR<Return_T,Int,LInt,parQ> Op() const
            {
                std::string tag = ClassName()+"::Op"
                    + "<" + ToString(op)
                    + "," + TypeName<Return_T>
                    + ">";
                
                TOOLS_PTIMER(timer,tag);
                
                if( !this->WellFormedQ() )
                {
                    return MatrixCSR<Return_T,Int,LInt,parQ> ( m, n, 0, 1 );
                }
                
                if constexpr (
                    (op == Op::Id)
                    ||
                    (Scalar::RealQ<Scal> && ( (op == Op::Conj) || (op == Op::Re) ) )
                )
                {
                    return MatrixCSR<Return_T,Int,LInt,parQ>( *this );
                }
                else if constexpr ( Scalar::RealQ<Scal> && (op == Op::Im) )
                {
                    if constexpr ( NotTransposedQ(op) )
                    {
                        return MatrixCSR<Return_T,Int,LInt,parQ> ( m, n, 0, 1 );
                    }
                    else
                    {
                        return MatrixCSR<Return_T,Int,LInt,parQ> ( n, m, 0, 1 );
                    }
                }
                else if constexpr ( NotTransposedQ(op) )
                {
                    MatrixCSR<Return_T,Int,LInt,parQ> B ( m, n, outer[m], thread_count );
                    
                    B.Outer().Read( outer.data(), thread_count );
                    B.Inner().Read( inner.data(), thread_count );
                    
                    cptr<Return_T> A_v = values;
                    mptr<Return_T> B_v = B.Values();
                    
                    Do<parQ>(
                        [A_v,B_v,this]( LInt k )
                        {
                            B_v[k] = Scalar::Op<op>(A_v[k]);
                        },
                        outer[m], thread_count
                    );
                    
                    return B;
                }
                else // if constexpr ( TransposedQ(op) )
                {
                    RequireJobPtr();
                    
                    Tensor2<LInt,Int> counters = CreateTransposeCounters();
                    
                    MatrixCSR<Return_T,Int,LInt,parQ> B ( n, m, outer[m], thread_count );

                    copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );

                    Do<parQ>(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c = counters.data(thread);
                            mptr<Int > B_i = B.Inner().data();
                            mptr<Scal> B_v = B.Value().data();
                            cptr<LInt> A_o = Outer().data();
                            cptr<Int > A_i = Inner().data();
                            cptr<Scal> A_v = Value().data();
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt k_begin = A_o[i  ];
                                const LInt k_end   = A_o[i+1];
                                
                                for( LInt k = k_end; k --> k_begin; )
                                {
                                    const Int j = A_i[k];
                                    const LInt pos = --c[j];
                                    B_i [pos] = i;
                                    B_v[pos] = Scalar::Op<op>(A_v[k]);
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
                    
            }
            
            MatrixCSR Transpose() const
            {
                return Op<Op::Trans>();
            }
            
            MatrixCSR ConjugateTranspose() const
            {
                return Op<Op::ConjTrans>();
            }
            
        public:
            
            // Supply an external list of values.
            template<typename A_T = Scal>
            void WriteDense( mptr<A_T> A, const Int ldA ) const
            {
                return this->WriteDenseValues( values.data(), A, ldA );
            }
            
            // Supply an external list of values.
            template<typename A_T = Scal>
            Tensor2<A_T,LInt> ToTensor2() const
            {
                return this->template ToTensor2WithValues<A_T>( values.data() );
            }
            
            
            void SortInner() const override
            {
                TOOLS_PTIMER(timer,ClassName()+"::SortInner");
                
                this->template SortInner_impl<false>(
                    outer.data(), inner.data(), values.data(), nullptr
                );
            }
            
            virtual void Compress() const override
            {
                // Removes duplicate {i,j}-pairs by adding their corresponding nonzero values.
                
                TOOLS_PTIMER(timer,MethodName("Compress"));

                Tensor1<LInt,LInt> C_outer;
                
                this->template Compress_impl<true,false>( outer, inner, values, C_outer );
            }
            
        public:
            
            std::tuple<
                Tensor1<LInt,Int>, Tensor1<Int,LInt>, Tensor1<Scal,LInt>,
                Assembler_T, Int, Int
            > Disband()
            {
                Tensor1<LInt, Int> outer_     = std::move(outer);
                Tensor1< Int,LInt> inner_     = std::move(inner);
                Tensor1<Scal,LInt> values_    = std::move(values);
                Assembler_T        assembler_ = std::move(assembler);
                Int m_ = m;
                Int n_ = n;
                
                *this = MatrixCSR();
                
                return { outer_, inner_, values_, assembler_, m_, n_ };
            }
            
            template<typename ExtScal>
            void Reassemble( cptr<ExtScal> unassemble_values )
            {
                TOOLS_PTIMER(timer,ClassName() + "::Reassemble");
                
                if( (assembler.Dimension(0) == LInt(0)) && (assembler.Dimension(1) == LInt(0))
                    )
                {
                    eprint(ClassName() + "::Reassemble" + "<" + TypeName<ExtScal> + ">: No assembler avaible. You have to construct the matrix with another constructor that allows to set the option assemblerQ = true.");
                }
                
                assembler.Dot(
                    Scal(1), unassemble_values, Scal(0), values.data()
                );
            }
            
        
//###########################################################
//####          Permute
//###########################################################
            
        public:
            
            MatrixCSR Permute(
                cref<Tensor1<Int,Int>> p,
                cref<Tensor1<Int,Int>> q,
                bool sortQ = true
            )
            {
                if( p.Dim(0) != m )
                {
                    eprint(MethodName("Permute")+": Length of first argument does not coincide with RowCount().");
                    return MatrixCSR();
                }
                
                if( q.Dim(0) != n )
                {
                    eprint(MethodName("Permute")+": Length of second argument does not coincide with ColCount().");
                    return MatrixCSR();
                }
                
                Permute( p.data(), q.data(), sortQ );
            }
            
            MatrixCSR Permute( cptr<Int> p, cptr<Int> q, bool sortQ = true )
            {
                if( p == nullptr )
                {
                    if( q == nullptr )
                    {
                        // Just make a copy.
                        return MatrixCSR(*this);
                    }
                    else
                    {
                        return PermuteCols(q,sortQ);
                    }
                }
                else
                {
                    if( q == nullptr )
                    {
                        return PermuteRows(p);
                    }
                    else
                    {
                        return PermuteRowsCols(p,q,sortQ);
                    }
                }
            }
            
        protected:
            
            MatrixCSR PermuteRows( cptr<Int> p )
            {
                MatrixCSR B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
                
                {
                    cptr<LInt> A_o = outer.data();
                    mptr<LInt> B_o = B.Outer().data();
                    
                    B_o[0] = 0;
                    
                    Do<parQ>(
                        [=,this]( const Int i )
                        {
                            const Int p_i = p[i];
                            B_o[i+1] = A_o[p_i+1] - A_o[p_i];
                        },
                        m, ThreadCount()
                    );
                }
                
                // TODO: Check what works better.
//                B.Outer().template Accumulate<parQ>( thread_count );
                B.Outer().Accumulate();
                
                {
                    auto & B_job_ptr = B.JobPtr();
                    
                    cptr<LInt> A_o = outer.data();
                    cptr<Int > A_i = inner.data();
                    mptr<Scal> A_v = values.data();
                    
                    cptr<LInt> B_o = B.Outer().data();
                    mptr<Int > B_i = B.Inner().data();
                    mptr<Scal> B_v = B.Values().data();
                    
                    Do<parQ>(
                        [=,this,&B]( const Int i )
                        {
                            const Int  p_i = p[i];
                            const LInt A_begin = A_o[p_i  ];
                            const LInt A_end   = A_o[p_i+1];
                            
                            const LInt B_begin = B_o[i  ];
                            
                            copy_buffer(&A_i[A_begin],&A_i[A_end],&B_i[B_begin]);
                            copy_buffer(&A_v[A_begin],&A_v[A_end],&B_v[B_begin]);
                            B.proven_inner_sortedQ = true;
                        },
                        B_job_ptr
                    );
                }
                
                return B;
            }
            
            MatrixCSR PermuteCols( cptr<Int> q, bool sortQ = true )
            {
                MatrixCSR B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
                
                Tensor1<Int,Int> q_inv_buffer ( ColCount() );
                
                mptr<Int> q_inv = q_inv_buffer.data();
                
                Do<parQ>(
                    [q,q_inv]( const Int j ) { q_inv[q[j]] = j; },
                    n, ThreadCount()
                );
                
                copy_buffer( outer.data(), B.Outer().data(), m+1 );
                
                Do<parQ>(
                    [=,this,&B]( const Int thread )
                    {
                        TwoArraySort<Int,Scal,LInt> S;
                        
//                        cptr<LInt> A_o = outer.data();
                        cptr<Int > A_i = inner.data();
                        cptr<Scal> A_v = values.data();
                        
                        cptr<LInt> B_o = B.Outer().data();
                        mptr<Int > B_i = B.Inner().data();
                        mptr<Scal> B_v = B.Values().data();
                        
                        const Int i_begin = B.JobPtr()[thread  ];
                        const Int i_end   = B.JobPtr()[thread+1];
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt B_begin = B_o[i  ];
                            const LInt B_end   = B_o[i+1];;
                            
                            for( LInt k = B_begin; k < B_begin; ++k )
                            {
                                B_i[B_begin] = q_inv[A_i[B_begin]];
                            }
                            
                            const LInt k_max = B_end - B_begin;
                            
                            copy_buffer(&A_v[B_begin],&B_v[B_begin],k_max);
                            
                            if( sortQ )
                            {
                                S(&B_i[B_begin],&B_v[B_begin],k_max);
                                B.proven_inner_sortedQ = true;
                            }
                        }
                    },
                    B.JobPtr().ThreadCount(), B.JobPtr().ThreadCount()
                );
                
                return B;
            }
            
            MatrixCSR PermuteRowsCols( cptr<Int> p, cptr<Int> q, bool sortQ = true )
            {
                MatrixCSR B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
                
                Tensor1<Int,Int> q_inv_buffer ( ColCount() );
                mptr<Int> q_inv = q_inv_buffer.data();
                
                Do<parQ>(
                    [q,q_inv]( const Int j ) { q_inv[q[j]] = j; },
                    n, ThreadCount()
                );
                
                {
                    cptr<LInt> A_o = outer.data();
                    mptr<LInt> B_o = B.Outer().data();
                    
                    B_o[0] = 0;
                    
                    Do<parQ>(
                        [=,this]( const Int i )
                        {
                            const Int p_i = p[i];
                            B_o[i+1] = A_o[p_i+1] - A_o[p_i];
                        },
                        m, ThreadCount()
                    );
                }
                
                // TODO: Check what works better.
//                B.Outer().template Accumulate<parQ>( thread_count );
                B.Outer().Accumulate();
                
                Do<parQ>(
                    [=,this,&B]( const Int thread )
                    {
                        cptr<LInt> A_o = outer.data();
                        cptr<Int > A_i = inner.data();
                        cptr<Scal> A_v = values.data();
                        
                        cptr<LInt> B_o = B.Outer().data();
                        mptr<Int > B_i = B.Inner().data();
                        mptr<Scal> B_v = B.Values().data();
                        
                        TwoArraySort<Int,Scal,LInt> S;
                        
                        const Int i_begin = B.JobPtr()[thread  ];
                        const Int i_end   = B.JobPtr()[thread+1];
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const Int p_i = p[i];
                            const LInt A_begin = A_o[p_i  ];
                            
                            const LInt B_begin = B_o[i  ];
                            const LInt B_end   = B_o[i+1];
                            
                            const LInt k_max = B_end - B_begin;
                            
                            for( LInt k = 0; k < k_max; ++k )
                            {
                                B_i[B_begin+k] = q_inv[A_i[A_begin+k]];
                            }
                            
                            copy_buffer(&A_v[A_begin],&B_v[B_begin],k_max);
                            
                            if( sortQ )
                            {
                                S(&B_i[B_begin],&B_v[B_begin],k_max );
                                B.proven_inner_sortedQ = true;
                            }
                        }
                    },
                    B.JobPtr().ThreadCount(), B.JobPtr().ThreadCount()
                );
                
                return B;
            }
            
        public:
            
            MatrixCSR Dot( const MatrixCSR & B ) const
            {
                TOOLS_PTIMER(timer,MethodName("Dot"));
                
                if( !this->WellFormedQ() )
                {
                    eprint(MethodName("Dot")+": Matrix is not well-formed.");
                    return MatrixCSR ();
                }

                RequireJobPtr();
                
                Tensor2<LInt,Int> counters ( thread_count, m, LInt(0) );
                
                // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                // https://en.wikipedia.org/wiki/Counting_sort
                
                Do<parQ>(
                    [=,this,&B,&counters]( const Int thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        mptr<LInt> c = counters.data(thread);
                        
                        cptr<LInt> A_o  = Outer().data();
                        cptr<Int>  A_i  = Inner().data();
                        
                        cptr<LInt> B_o  = B.Outer().data();
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            LInt c_i = 0;
                            
                            const LInt jj_begin = A_o[i  ];
                            const LInt jj_end   = A_o[i+1];
                            
                            for( LInt jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const Int j = A_i[jj];
                                
                                c_i += (B_o[j+1] - B_o[j]);
                            }
                            
                            c[i] = c_i;
                        }
                    },
                    thread_count, thread_count
                );
                
                AccumulateAssemblyCounters<parQ>( counters );
                
                const LInt nnz = counters.data(thread_count-1)[m-1];
                
                MatrixCSR C ( m, B.ColCount(), nnz, thread_count );
                
                copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                
                Do<parQ>(
                    [&,this]( const Int thread )
                    {
                        const Int i_begin   = job_ptr[thread  ];
                        const Int i_end     = job_ptr[thread+1];
                        
                        mptr<LInt> c        = counters.data(thread);
                        
                        cptr<LInt> A_o  = Outer().data();
                        cptr< Int> A_i  = Inner().data();
                        cptr<Scal> A_v  = Value().data();
                        
                        cptr<LInt> B_o  = B.Outer().data();
                        cptr< Int> B_i  = B.Inner().data();
                        cptr<Scal> B_v = B.Value().data();
                        
                        mptr< Int> C_inner  = C.Inner().data();
                        mptr<Scal> C_values = C.Value().data();
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt jj_begin = A_o[i  ];
                            const LInt jj_end   = A_o[i+1];
                            
                            for( LInt jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const Int j = A_i[jj];
                                
                                const LInt kk_begin = B_o[j  ];
                                const LInt kk_end   = B_o[j+1];
                                
                                for( LInt kk = kk_end; kk --> kk_begin; )
                                {
                                    const Int k = B_i[kk];
                                    const LInt pos = --c[i];
                                    
                                    C_inner [pos] = k;
                                    C_values[pos] = A_v[jj] * B_v[kk];
                                }
                            }
                        }
                    },
                    thread_count, thread_count
                );
                
                // Finished expansion phase (counting sort).
                
                // Finally we row-sort inner and compressQ duplicates in inner and values.
                C.Compress();
                
                return C;
            }
            
            Sparse::BinaryMatrixCSR<Int,LInt,parQ> DotBinary( const MatrixCSR & B ) const
            {
                Sparse::BinaryMatrixCSR<Int,LInt,parQ> result;
                
                Base_T C = this->DotBinary_(B);
                
                swap(result,C);
                
                return result;
            }
            
            
        public:
            
//###########################################################
//####          Matrix Multiplication
//###########################################################
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X, const Int ldX,
                const b_T beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( values.data(), alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X,
                const b_T beta,  mptr<Y_T> Y,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( values.data(), alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            // Use own nonzero values.
            template<typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m )
                {
                    const Int nrhs = 1;
                    
                    this->template Dot_<1>( values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
            }
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor2<X_T,Int>> X,
                const b_T beta,  mref<Tensor2<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m && (X.Dim(1) == Y.Dim(1)) )
                {
                    const Int nrhs = X.Dim(1);
                    
                    this->template Dot_<NRHS>( values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
            }
            
            
            
            // Use external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cptr<T_ext> ext_values,
                const a_T alpha, cptr<T_ext> X, const Int ldX,
                const b_T beta,  mptr<T_ext> Y, const Int ldY,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Use external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cptr<T_ext> ext_values,
                const a_T alpha, cptr<T_ext> X,
                const b_T beta,  mptr<T_ext> Y,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            template<typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cref<Tensor1<T_ext,Int>> ext_values,
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m )
                {
                    const Int nrhs = 1;
                    
                    this->template Dot_<1>( ext_values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = VarSize, typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cref<Tensor1<T_ext,Int>> ext_values,
                const a_T alpha, cref<Tensor2<X_T,Int>> X,
                const b_T beta,  mref<Tensor2<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m && (X.Dim(1) == Y.Dim(1)) )
                {
                    const Int nrhs = X.Dim(1);
                    
                    this->template Dot_<NRHS>( ext_values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(MethodName("Dot")+": shapes of matrix, input, and output do not match.");
                }
            }
            
            void FillLowerTriangleFromUpperTriangle()
            {
                FillLowerTriangleFromUpperTriangle( values.data() );
            }
            
            void FillUpperTriangleFromLowerTriangle()
            {
                FillUpperTriangleFromLowerTriangle( values.data() );
            }
            
            
        public:
            
            
            void LoadFromFile(
                const std::filesystem::path & file, const Int thread_count_
            )
            {
                std::string tag = MethodName("LoadFromMatrixMarket");
                
                TOOLS_PTIMER(timer,tag);
                
                std::ifstream  s ( file );
        
                if( !s.good() )
                {
                    eprint(tag + ": File " + file.string() + " not found. Aborting.");
                    return;
                }
                
                Int m;
                Int n;
                Int nnz;
                
                s >> m;
                s >> n;
                s >> nnz;
                
                MatrixCSR A( m, n, nnz, thread_count_ );
                
                mptr<LInt> rp = A.Outer().data();
                for( Int i = 0; i < n+1; ++i )
                {
                    s >> rp[i];
                }
                
                mptr<Int> ci = A.Inner().data();
                for( Int i = 0; i < nnz; ++i )
                {
                    s >> ci[i];
                }
                
                mptr<Scal> a = A.Values().data();
                for( Int i = 0; i < nnz; ++i )
                {
                    s >> a[i];
                }
                
                swap( A, *this);
            }
            
            
            bool SymmetricQ() const
            {
                // This routine is not optimized.
                // Only meant for debugging.
                
                bool symmetricQ = true;
                
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
                            const Scal A_ij = values[k];
                            const Scal A_ji = this->operator()(j,i);

                            const bool foundQ = (A_ij == A_ji);
                            
                            symmetricQ = symmetricQ && foundQ;
                            
                            unmatched_count += (!foundQ);
                        }
                    }
                }
                
                if( !symmetricQ )
                {
                    TOOLS_DUMP(unmatched_count);
                }
                
                return symmetricQ;
            }
            
            static MatrixCSR IdentityMatrix( const Int n, const Int thread_count = 1 )
            {
                MatrixCSR A ( n, n, n, thread_count );
                A.Outer().iota();
                A.Inner().iota();
                A.Value().Fill(Scal(1));
                
                return A;
            }
            
            std::string Stats() const
            {
                return std::string()
                + "\n==== "+ClassName()+" Stats ====" + "\n\n"
                + " RowCount()      = " + ToString(RowCount()) + "\n"
                + " ColCount()      = " + ToString(ColCount()) + "\n"
                + " NonzeroCount()  = " + ToString(NonzeroCount()) + "\n"
                + " ThreadCount()   = " + ToString(ThreadCount()) + "\n"
                + " Outer().Size()  = " + ToString(Outer().Size()) + "\n"
                + " Inner().Size()  = " + ToString(Inner().Size()) + "\n"
                + " Value().Size()  = " + ToString(Value().Size()) + "\n"
                + "\n==== "+ClassName()+" Stats ====\n\n";
            }


#include "MatrixCSR/SortInner.hpp"
#include "MatrixCSR/FromTriples.hpp"
#include "MatrixCSR/MatrixMarket.hpp"
            
        public:
            
            static std::string MethodName( const std::string & tag )
            {
                return ClassName() + "::" + tag;
            }
            
            static std::string ClassName()
            {
                return std::string("Sparse::MatrixCSR")
                    + "<" + TypeName<Scal>
                    + "," + TypeName<Int>
                    + "," + TypeName<LInt>
                    + "," + ToString(parQ)
                    + ">";
            }
            
        }; // MatrixCSR
    
        
    } // namespace Sparse
    
} // namespace Tensors
