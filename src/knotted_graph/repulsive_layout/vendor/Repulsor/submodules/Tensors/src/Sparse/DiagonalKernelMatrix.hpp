#pragma once

namespace Tensors
{
    template<typename Kernel_T>
    class DiagonalKernelMatrix final
    {
    public:
        
        using Scal     = typename Kernel_T::Scal;
        using Int      = typename Kernel_T::Int;
        using LInt     = typename Kernel_T::LInt;
        using Scal_in  = typename Kernel_T::Scal_in;
        using Scal_out = typename Kernel_T::Scal_out;

        
        DiagonalKernelMatrix(
              const Int n_,
              const Int thread_count_
              )
        :   n ( n_ )
        ,   thread_count(thread_count_)
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_NRHS }
        {}
        
        // Default constructor
        DiagonalKernelMatrix()
        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_NRHS }
        {}
        
        // Destructor
        ~DiagonalKernelMatrix() = default;
        // Copy constructor
        DiagonalKernelMatrix( const DiagonalKernelMatrix & other ) = default;
        // Copy assignment operator
        DiagonalKernelMatrix & operator=( const DiagonalKernelMatrix & other ) = default;
        // Move constructor
        DiagonalKernelMatrix( DiagonalKernelMatrix && other ) = default;
        // Move assignment operator
        DiagonalKernelMatrix & operator=( DiagonalKernelMatrix && other ) = default;
        
    protected:
        
        const Int n            = 0;
        const Int thread_count = 1;
        
        Kernel_T kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_NRHS };
        
    public:
        
        Int RowCount() const
        {
            return n * Kernel_T::RowCount();
        }
        
        Int ColCount() const
        {
            return n * Kernel_T::ColCount();
        }
        
        Int NonzeroCount() const
        {
            return n * Kernel_T::BLOCK_NNZ;
        }
        
        
//###########################################################
//      Symmetrization
//###########################################################
        
    public:
        
        
//###########################################################
//      Matrix multiplication
//###########################################################
        
        
        void Scale( mptr<Scal_out> Y, cref<Scal_out> beta, const Int nrhs ) const
        {
            const Int size = RowCount() * nrhs;
            
            if( beta == static_cast<Scal_out>(0) )
            {
                zerofy_buffer<VarSize,Parallel>( Y, size, thread_count );
            }
            else
            {
                scale_buffer<VarSize,Parallel>( beta, Y, size, thread_count );
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
            
            if( (alpha == static_cast<Scal_out>(0)) || (NonzeroCount() <= 0) )
            {
                Scale( Y, beta, nrhs );
                return;
            }
            
            const auto & job_ptr = JobPointers<Int>(n,thread_count);
            
            ParallelDo(
                [&job_ptr,A,alpha,X,beta,Y,nrhs]( const Int thread)
                {
                    // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                    Kernel_T ker ( A, alpha, X, beta, Y, nrhs );
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        ker.CleanseY();
                        ker.ApplyBlock(i,i);
                        ker.WriteY(i);
                    }
                },
                thread_count
            );
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Sparse::DiagonalKernelMatrix<"+kernel.ClassName()+">";
        }
        
    };
    
} // namespace Tensors
