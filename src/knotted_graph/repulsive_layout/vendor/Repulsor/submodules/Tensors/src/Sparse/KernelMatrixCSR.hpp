#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Kernel_T>
        class KernelMatrixCSR final
        {
        public:
            
            using Scal     = typename Kernel_T::Scal;
            using Int      = typename Kernel_T::Int;
            using LInt     = typename Kernel_T::LInt;
            using Scal_in  = typename Kernel_T::Scal_in;
            using Scal_out = typename Kernel_T::Scal_out;
            
            static constexpr Parallel_T parQ = Parallel;
            
            using Pattern_T = Sparse::PatternCSR<Int,LInt,parQ>;
            
            //        KernelMatrixCSR()
            //        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_NRHS }
            //        {}
            
            explicit KernelMatrixCSR( const Pattern_T & pattern_ )
            :   pattern ( pattern_ )
            ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_NRHS }
            {}
            
            // No default constructor
            KernelMatrixCSR() = delete;
            // Destructor
            ~KernelMatrixCSR() = default;
            // Copy constructor
            KernelMatrixCSR( const KernelMatrixCSR & other ) = default;
            // Copy assignment operator
            KernelMatrixCSR & operator=( const KernelMatrixCSR & other ) = default;
            // Move constructor
            KernelMatrixCSR( KernelMatrixCSR && other ) = default;
            // Move assignment operator
            KernelMatrixCSR & operator=( KernelMatrixCSR && other ) = default;
            
        protected:
            
            const Pattern_T & pattern;
            Kernel_T kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_NRHS };
            
        public:
            
            Int RowCount() const
            {
                return pattern.RowCount() * Kernel_T::RowCount();
            }
            
            Int ColCount() const
            {
                return pattern.ColCount() * Kernel_T::ColCount();
            }
            
            LInt NonzeroCount() const
            {
                return pattern.NonzeroCount() * Kernel_T::BLOCK_NNZ;
            }
            
            
//###########################################################
//      Symmetrization
//###########################################################
            
        public:
            
            void FillLowerTriangleFromUpperTriangle( mptr<Scal> values ) const
            {
                TOOLS_PTIMER(timer,ClassName()+"::FillLowerTriangleFromUpperTriangle");
                
                if( pattern.WellFormedQ() && (pattern.RowCount()>= pattern.ColCount()) )
                {
                    cptr<LInt> diag   = pattern.Diag().data();
                    cptr<LInt> outer  = pattern.Outer().data();
                    cptr<Int>  inner  = pattern.Inner().data();
                    
                    const auto & job_ptr = pattern.LowerTriangularJobPtr();
                    
                    const Int thread_count = job_ptr.ThreadCount();
                    
                    ParallelDo(
                        [&job_ptr,outer,inner,values,diag]( const Int thread )
                        {
                            Kernel_T ker ( values );
                            
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt k_begin = outer[i];
                                const LInt k_end   =  diag[i];
                                
                                for( LInt k = k_begin; k < k_end; ++k )
                                {
                                    const Int j = inner[k];
                                    
                                    LInt L =  diag[j];
                                    LInt R = outer[j+1]-1;
                                    
                                    while( L < R )
                                    {
                                        const LInt M = R - (R-L)/LInt(2);
                                        const  Int col = inner[M];
                                        
                                        if( col > i )
                                        {
                                            R = M-1;
                                        }
                                        else
                                        {
                                            L = M;
                                        }
                                    }
                                    
                                    ker.TransposeBlock(L,k);
                                    
                                } // for( LInt k = k_begin; k < k_end; ++k )
                                
                            } // for( Int i = i_begin; i < i_end; ++i )
                        },
                        thread_count
                    );
                    
                }
            }
            
            
//###########################################################
//      Matrix multiplication
//###########################################################
            
            void Scale( mptr<Scal_out> Y, cref<Scal_out> beta, const Int nrhs ) const
            {
                const Int size = RowCount() * nrhs;
                
                if( beta == static_cast<Scal_out>(0) )
                {
                    zerofy_buffer( Y, size, pattern.ThreadCount() );
                }
                else
                {
                    scale_buffer(beta, Y, size, pattern.ThreadCount() );
                }
            }
            
            TOOLS_FORCE_FLATTENING void Dot(
                cptr<Scal> A,
                cref<Scal_out> alpha, cptr<Scal_in>  X,
                cref<Scal_out> beta,  mptr<Scal_out> Y,
                const Int nrhs
            ) const
            {
                TOOLS_PTIMER(timer,ClassName()+"::Dot" );
                
                if( (alpha == static_cast<Scal_out>(0)) || (NonzeroCount() <= Int(0)) )
                {
                    Scale( Y, beta, nrhs );
                    return;
                }
                
                const auto & job_ptr = pattern.JobPtr();
                
                const Int thread_count = job_ptr.ThreadCount();
                
                ParallelDo(
                    [=, &job_ptr, this]( const Int thread )
                    {
                        // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                        Kernel_T ker ( A, alpha, X, beta, Y, nrhs );
                        
                        cptr<LInt> rp = pattern.Outer().data();
                        cptr< Int> ci = pattern.Inner().data();
                        
                        // Kernel is supposed the following rows of pattern:
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        const LInt last_k = rp[i_end];
                        
                        constexpr LInt look_ahead = 1;
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            // These are the corresponding nonzero blocks in i-th row.
                            const LInt k_begin = rp[i  ];
                            const LInt k_end   = rp[i+1];
                            
                            if( k_end > k_begin )
                            {
                                // Clear the local vector chunk of the kernel.
                                ker.CleanseY();

                                for( LInt k = k_begin; k < k_end; ++k )
                                {
                                    const Int j = ci[k];
                                    
                                    if ( k + look_ahead < last_k )
                                    {
                                        ker.Prefetch(k,ci[k+look_ahead]);
                                    }
                                    
                                    // Let the kernel apply to the k-th block to the j-th chunk of the input.
                                    // The result is stored in the kernel's local matrix chunk of X.
                                    ker.ApplyBlock(k,j);
                                }

                                // Incorporate the kernel's local matrix chunk into the i-th chunk in the output Y.
                                ker.WriteY(i);
                            }
                            else
                            {
                                // Make sure that Y(i) is correctly overwritten in the case that there are not entries in the row.
                                ker.WriteYZero(i);
                            }
                            
                        }
                    },
                    thread_count
                );
            }
            
//###########################################################
//####          Permute
//###########################################################
            
        public:
            
//  This is outdated as of now. Will have to be fixed some day if need.
//
//            [[nodiscard]] Permutation<LInt> Permute(
//                const Permutation<Int> & p,  // row    permutation
//                const Permutation<Int> & q  // column permutation
//            )
//            {
//                // Modifies inner, outer, and values accordingly; returns the permutation to be applied to the nonzero values.
//                
//                const LInt nnz = this->inner.Size();
//                
//                Permutation<LInt> perm;
//                
//                std::tie( this->outer, this->inner, perm ) = SparseMatrixPermutation(
//                    this->outer.data(), this->inner.data(), p, q, nnz, true
//                );
//
//                if( !p.TrivialQ() || !q.TrivialQ() )
//                {
//                    cptr<Scal> u = this->new_values.data();
//                    mptr<Scal> v = this->values.data();
//                    Tensor1<Scal,LInt> new_values ( nnz );
//                    
//                    Do<parQ>(
//                        [&]( const Int i ) { v[i] = u[perm[i]]; },
//                        nnz, this->ThreadCount()
//                    );
//                    
//                    swap( this->values, new_values);
//                }
//                
//                this->proven_inner_sortedQ = true;
//                
//                this->diag_ptr_initialized = false;
//                this->job_ptr_initialized  = false;
//                this->upper_triangular_job_ptr_initialized = false;
//                this->lower_triangular_job_ptr_initialized = false;
//                
//                return perm;
//            }

        public:
            
            [[nodiscard]] std::string ClassName() const
            {
                return std::string("Sparse::KernelMatrixCSR")
                + "<" + kernel.ClassName()
                + "," + ToString(parQ)
                + ">";
            }
            
        }; // class KernelMatrixCSR
        
    }; // namespace Sparse
        
} // namespace Tensors
