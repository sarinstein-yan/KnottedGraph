#pragma once

namespace Tensors
{
    template<
        Size_T rows_ct_,   Size_T cols_ct_,
        Size_T m_default_, Size_T n_default_,
        Op store_op, Op op,
        typename Scal_, IntQ Int_
    >
    class MatrixBlockMajor final
    {
        /// TODO: Data layout
        /// We are using a `Tensor2<Tiny::Matrix<...>,Int>` at the moment.
        /// It might be be better to use a plain `Tensor1<Scal,Int>` with pointers.
        /// This way, we can call `Read` and `Write` with various `Op` modes after the
        /// class has been instantiated.
        /// Alas, we might lose some benefits,
        /// when `rows_ct > VarSize` or `cols_ct > VarSize`.
        /// But I guess these benefits are tiny or not existing.

        
        /// TODO: Separate real and imaginary parts if `ComplexQ<Scal>`.
        /// This should improve the vectorization in BLAS3 kernels.
        
    public:
        
        using Scal = Scal_;
        using Int  = Int_;
        
        static constexpr Int rows_ct = rows_ct_;
        static constexpr Int cols_ct = cols_ct_;
        static constexpr Int m_default = m_default_;
        static constexpr Int n_default = n_default_;
        
        static_assert(
            (store_op == Op::Id) || (store_op == Op::Trans),
            "Only the storage operators Op::Id and Op::Trans are allowed."
        );
        
    private:
        
        static constexpr Int m_threshold = 1;
        static constexpr Int n_threshold = 1;
        
        // If any of the dimensions is prescribed
        
        static constexpr bool rows_fixed_Q = (VarSize < rows_ct);
        static constexpr bool cols_fixed_Q = (VarSize < cols_ct);
        
        static constexpr Int M_ct = TransposedQ(op) ? cols_ct : rows_ct;
        static constexpr Int N_ct = TransposedQ(op) ? rows_ct : cols_ct;
        
        static constexpr bool M_fixed_Q = (VarSize < M_ct);
        static constexpr bool N_fixed_Q = (VarSize < N_ct);
        
        static constexpr bool M_small_Q = M_fixed_Q && (M_ct <= m_threshold);
        static constexpr bool N_small_Q = N_fixed_Q && (N_ct <= n_threshold);
        
    public:
        
        static constexpr Int m = M_small_Q ? M_ct : m_default;
        static constexpr Int n = N_small_Q ? N_ct : n_default;
        
        static constexpr Int block_size = m * n;
        
    private:
        
        static constexpr Int rows_block_count_ct         = NotTransposedQ(op)       ? CeilDivide (rows_ct,m)   : CeilDivide (cols_ct,n)   ;
        static constexpr Int cols_block_count_ct         = NotTransposedQ(op)       ? CeilDivide (cols_ct,n)   : CeilDivide (rows_ct,m)   ;
        static constexpr Int rows_full_block_count_ct    = NotTransposedQ(op)       ? FloorDivide(rows_ct,m)   : FloorDivide(cols_ct,n)   ;
        static constexpr Int cols_full_block_count_ct    = NotTransposedQ(op)       ? FloorDivide(cols_ct,n)   : FloorDivide(rows_ct,m)   ;
        static constexpr Int rows_reg_ct                 = NotTransposedQ(op)       ? m * rows_block_count_ct  : n * cols_block_count_ct  ;
        static constexpr Int cols_reg_ct                 = NotTransposedQ(op)       ? n * cols_block_count_ct  : m * rows_block_count_ct  ;
        static constexpr Int rows_tail_ct                = rows_ct - rows_reg_ct ;
        static constexpr Int cols_tail_ct                = cols_ct - cols_reg_ct ;
        static constexpr Int m_tail_ct                   = NotTransposedQ(op)       ? rows_tail_ct             : cols_tail_ct             ;
        static constexpr Int n_tail_ct                   = NotTransposedQ(op)       ? cols_tail_ct             : rows_tail_ct             ;
    
        static constexpr Int M_block_count_ct            = NotTransposedQ(store_op) ? rows_block_count_ct      : cols_block_count_ct      ;
        static constexpr Int N_block_count_ct            = NotTransposedQ(store_op) ? cols_block_count_ct      : rows_block_count_ct      ;
        static constexpr Int M_full_block_count_ct       = NotTransposedQ(store_op) ? rows_full_block_count_ct : cols_full_block_count_ct ;
        static constexpr Int N_full_block_count_ct       = NotTransposedQ(store_op) ? cols_full_block_count_ct : rows_full_block_count_ct ;
        
    public:
        
        using Block_T = Tiny::Matrix<m,n,Scal,Int>;
    
        
    private:
        
        // Sizes at runtime.

        const Int rows_rt;
        const Int cols_rt;
        
        const Int rows_block_count_rt;
        const Int cols_block_count_rt;
        
        const Int rows_full_block_count_rt;
        const Int cols_full_block_count_rt;
        
        const Int rows_reg_rt;
        const Int cols_reg_rt;
        
        const Int rows_tail_rt;
        const Int cols_tail_rt;
        
        const Int m_tail_rt;
        const Int n_tail_rt;
        
//        const Int M_rt;
//        const Int N_rt;
        
        const Int thread_count = 1;
        
        const Int M_block_count_rt;
        const Int N_block_count_rt;
        
        const Int M_full_block_count_rt;
        const Int N_full_block_count_rt;
        
        const Int M_rt;
        const Int N_rt;
//
//        const Int M_tail_rt;
//        const Int N_tail_rt;
        
    public:
        
//        Tensor1<Scal,Int> blocks;
            
        Tensor3<Scal,Int> blocks;
        
    public:
        
        MatrixBlockMajor() = delete;
        
        
        /// Suffix `_rt` stands for "run time".
        /// Suffix `_ct` stands for "compile time".

        /// `blocks` is an array of size `M_block_count_rt` x `N_block_count_rt`
        /// containing tiny `m` x `n` blocks.

        /// The aim is to convert between
        /// a matrix A of size `rows_rt` x `cols_rt` of size `cols_rt` x `rows_rt`
        /// and the block representation `blocks`.
        ///
        /// More precisely:
        ///
        /// If `NotTransposedQ(op)`, then `A` is a matrix of
        /// size `rows_rt` x `cols_rt `, and it is supposed to be decomposed into
        /// blocks of size `m` x `n`.
        /// Otherwise it size of size `cols_rt` x `rows_rt `,
        /// and it will be decomposed into blocks of size `n` x `m`
        
        /// We decompose
        ///
        /// `rows_rt = m * cols_block_count_rt + M_tail_rt`
        /// `cols_rt = n * N_block_count_rt + N_tail_rt`
        
        MatrixBlockMajor( const Int rows_rt_ = rows_ct, const Int cols_rt_ = cols_ct, const Int thread_count_ = 1 )
        :   rows_rt                     ( rows_rt_ )
        ,   cols_rt                     ( cols_rt_ )
        ,   rows_block_count_rt         ( NotTransposedQ(op)       ? CeilDivide (rows_rt,m)   : CeilDivide (rows_rt,n)   )
        ,   cols_block_count_rt         ( NotTransposedQ(op)       ? CeilDivide (cols_rt,n)   : CeilDivide (cols_rt,m)   )
        ,   rows_full_block_count_rt    ( NotTransposedQ(op)       ? FloorDivide(rows_rt,m)   : FloorDivide(rows_rt,n)   )
        ,   cols_full_block_count_rt    ( NotTransposedQ(op)       ? FloorDivide(cols_rt,n)   : FloorDivide(cols_rt,m)   )
        ,   rows_reg_rt                 ( NotTransposedQ(op)       ? m * rows_block_count_rt  : n * rows_block_count_rt  )
        ,   cols_reg_rt                 ( NotTransposedQ(op)       ? n * cols_block_count_rt  : m * cols_block_count_rt  )
        ,   rows_tail_rt                ( rows_rt - rows_reg_rt )
        ,   cols_tail_rt                ( cols_rt - cols_reg_rt )
        ,   m_tail_rt                   ( NotTransposedQ(op)       ? rows_tail_rt             : cols_tail_rt             )
        ,   n_tail_rt                   ( NotTransposedQ(op)       ? cols_tail_rt             : rows_tail_rt             )
        ,   thread_count                ( thread_count_ )
        ,   M_block_count_rt            ( NotTransposedQ(store_op) ? rows_block_count_rt      : cols_block_count_rt      )
        ,   N_block_count_rt            ( NotTransposedQ(store_op) ? cols_block_count_rt      : rows_block_count_rt      )
        ,   M_full_block_count_rt       ( NotTransposedQ(store_op) ? rows_full_block_count_rt : cols_full_block_count_rt )
        ,   N_full_block_count_rt       ( NotTransposedQ(store_op) ? cols_full_block_count_rt : rows_full_block_count_rt )
        ,   M_rt                        ( m * M_block_count_rt )
        ,   N_rt                        ( n * N_block_count_rt )
//        ,   blocks                      ( M_rt * N_rt )
        ,  blocks                       ( M_block_count_rt, N_block_count_rt, m * n)
        {
            
        }
        
        ~MatrixBlockMajor() = default;
        
    public:
        
        
//        mptr<Block_T> operator[]( const Int M_blk )
//        {
//            return blocks[M_blk];
//        }
//        
//        cptr<Block_T> operator[]( const Int M_blk ) const
//        {
//            return blocks[M_blk];
//        }
//        
//        mref<Block_T> operator()( const Int M_blk, const Int N_blk )
//        {
//            return blocks(M_blk,N_blk);
//        }
//        
//        cref<Block_T> operator()( const Int M_blk, const Int N_blk ) const
//        {
//            return blocks(M_blk,N_blk);
//        }
            
        
        mptr<Scal> data( const Int M_blk, const Int N_blk )
        {
//            return blocks.data( N_rt * m * M_blk + n * N_blk );
            return blocks.data( M_blk, N_blk );
        }
            
        cptr<Scal> data( const Int M_blk, const Int N_blk ) const
        {
//            return blocks.data( N_rt * m * M_blk + n * N_blk );
            return blocks.data( M_blk, N_blk );
        }
        
        cptr<Scal> operator()( const Int M_blk, const Int N_blk ) const
        {
//            return blocks.data( N_rt * m * M_blk + n * N_blk );
            return blocks.data( M_blk, N_blk );
        }

        Int RowCount() const
        {
            if constexpr ( rows_fixed_Q )
            {
                return rows_ct;
            }
            else
            {
                return rows_rt;
            }
        }
        
        Int ColCount() const
        {
            if constexpr ( cols_fixed_Q )
            {
                return cols_ct;
            }
            else
            {
                return cols_rt;
            }
        }
        
        Int M_FullBlockCount() const
        {
            if constexpr ( M_fixed_Q )
            {
                return M_full_block_count_ct;
            }
            else
            {
                return M_full_block_count_rt;
            }
        }
        
        Int N_FullBlockCount() const
        {
            if constexpr ( N_fixed_Q )
            {
                return N_full_block_count_ct;
            }
            else
            {
                return N_full_block_count_rt;
            }
        }
        
        Int M_BlockCount() const
        {
            if constexpr ( M_fixed_Q )
            {
                return M_block_count_ct;
            }
            else
            {
                return M_block_count_rt;
            }
        }
        
        Int N_BlockCount() const
        {
            if constexpr ( N_fixed_Q )
            {
                return N_block_count_ct;
            }
            else
            {
                return N_block_count_rt;
            }
        }

        Int RowTailCount() const
        {
            if constexpr ( rows_fixed_Q )
            {
                return rows_tail_ct;
            }
            else
            {
                return rows_tail_rt;
            }
        }

        Int ColTailCount() const
        {
            if constexpr ( cols_fixed_Q )
            {
                return cols_tail_ct;
            }
            else
            {
                return cols_tail_rt;
            }
        }
            
    public:
        
        template<typename A_T, typename A_I>
        void Read_naive( cptr<A_T> A, const A_I ldA_ )
        {
            const Int ldA  = static_cast<Int(ldA_);
            
            Do<parQ>(
                [=,this]( const Int thread )
                {
                    const Int M_blk_begin = JobPointer( M_BlockCount(), thread_count, thread     );
                    const Int M_blk_end   = JobPointer( M_BlockCount(), thread_count, thread + 1 );

                    for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                    {
                        for( Int N_blk = 0; N_blk < N_BlockCount(); ++N_blk )
                        {
                            Tiny::Matrix<m,n,Scal,Int> a;

                            a.template Read<Op::Id>( &A[ldA * m * M_blk + n * N_blk], ldA );

                            a.template Write<Op::Id>( blocks.data(M_blk,N_blk) );
                        }
                    }
                },
                thread_count, thread_count
            );
        }
            
    public:
        
        template<typename A_T, typename A_I>
        void Read( cptr<A_T> A, const A_I ldA_ )
        {
            if( ldA_ <= 0 )
            {
                eprint(ClassName()+"::Read<" + TypeName<A_T> + "," + TypeName<A_I> + ">: leading dimensions <= 0 ");
                return;
            }
            
            const Int ldA = static_cast<Int>(ldA_);
            
            if constexpr ( NotTransposedQ(store_op) == NotTransposedQ(op) )
            {
                Read_N( A, ldA );
            }
            else
            {
                Read_T( A, ldA );
            }
        }
    
    public:
            
        template<
            typename alpha_T, typename beta_T, typename C_T, typename C_I
        >
        void Write(
                const alpha_T alpha_, const beta_T beta_,
                mptr<C_T> C, const C_I ldC_
        ) const
        {
            if( ldC_ <= 0 )
            {
                eprint(ClassName()+"::Write<" + "," + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + "," + TypeName<C_I> + ">: leading dimension <= 0 ");

                return;
            }
            
            using F = Scalar::Flag;
            
            const F alpha_flag = Scalar::ToFlag(alpha_);
            const F beta_flag  = Scalar::ToFlag(beta_);
            
            const C_T alpha = scalar_cast<C_T>(alpha_);
            const C_T beta  = scalar_cast<C_T>(beta_);
            
            const Int ldC = static_cast<Int>(ldC_);
            

            if( alpha_flag == F::Plus)
            {
                if( beta_flag == F::Zero )
                {
                    Write_<F::Plus,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Plus )
                {
                    Write_<F::Plus,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Minus )
                {
                    Write_<F::Plus,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_<F::Plus,F::Generic>( alpha, beta, C, ldC );
                }
            }
            else if( alpha_flag == F::Zero )
            {
                if( beta_flag == F::Zero )
                {
                    Write_<F::Zero,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Plus )
                {
                    Write_<F::Zero,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Minus )
                {
                    Write_<F::Zero,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_<F::Zero,F::Generic>( alpha, beta, C, ldC );
                }
            }
            else if( alpha_flag == F::Minus )
            {
                if( beta_flag == F::Zero )
                {
                    Write_<F::Minus,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Plus )
                {
                    Write_<F::Minus,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Minus )
                {
                    Write_<F::Minus,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_<F::Minus,F::Generic>( alpha, beta, C, ldC );
                }
            }
            else
            {
                if( beta_flag == F::Zero )
                {
                    Write_<F::Generic,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Plus )
                {
                    Write_<F::Generic,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta_flag == F::Minus )
                {
                    Write_<F::Generic,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_<F::Generic,F::Generic>( alpha, beta, C, ldC );
                }
            }
        }
            
    private:
            
        template<
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename alpha_T, typename beta_T, typename C_T
        >
        void Write_(
                const alpha_T alpha, const beta_T beta,
                mptr<C_T> C, const Int ldC
        ) const
        {
            if constexpr ( NotTransposedQ(store_op) )
            {
                Write_N<alpha_flag,beta_flag>( alpha, beta, C, ldC );
            }
            else if constexpr ( TransposedQ(store_op) )
            {
                Write_T<alpha_flag,beta_flag>( alpha, beta, C, ldC );
            }
        }
            
        
        
private:
        

#include "MatrixBlockMajor/Read.hpp"
//#include "MatrixBlockMajor/Read_with_pointers.hpp"

        
#include "MatrixBlockMajor/Write.hpp"
//#include "MatrixBlockMajor/Write_with_pointers.hpp"
    public:
        
        [[nodiscard]] std::string friend ToString( cref<MatrixBlockMajor> M )
        {
            return ToString(M.blocks);
        }
        
        [[nodiscard]] std::string ClassName() const
        {
            return std::string("MatrixBlockMajor") +"<"
                + ToString(M_ct) +  "," + ToString(N_ct) + ","
                + ToString(m) +  "," + ToString(n) + ","
                + ToString(store_op) + "," + ToString(op) + ","
                + TypeName<Scal>
            + ">(" + ToString(RowCount()) + "," + ToString(ColCount()) + ")";
        }
        
    }; // class MatrixBlockMajor

}

