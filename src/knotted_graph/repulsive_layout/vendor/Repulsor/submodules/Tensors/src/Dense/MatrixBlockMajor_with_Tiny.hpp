#pragma once

namespace Tensors
{
    template<
        Size_T rows_ct,   Size_T cols_ct,
        Size_T m_default, Size_T n_default,
        Op store_op, Op op,
        typename Scal
    >
    class MatrixBlockMajor final
    {
        /// TODO: Data layout
        /// We are using a `Tensor2<Tiny::Matrix<...>,Size_T>` at the moment.
        /// It might be be better to use a plain `Tensor1<Scal,Size_T>` with pointers.
        /// This way, we can call `Read` and `Write` with various `Op` modes after the
        /// class has been instantiated.
        /// Alas, we might lose some benefits,
        /// when `rows_ct > VarSize` or `cols_ct > VarSize`.
        /// But I guess these benefits are tiny or not existing.

        
        /// TODO: Separate real and imaginary parts if `ComplexQ<Scal>`.
        /// This should improve the vectorization in BLAS3 kernels.
        
        static_assert(
            (store_op == Op::Id) || (store_op == Op::Trans),
            "Only the storage operators Op::Id and Op::Trans are allowed."
        );
        
    private:
        
        static constexpr Size_T m_threshold = 1;
        static constexpr Size_T n_threshold = 1;
        
        // If any of the dimensions is prescribed
        
        static constexpr bool rows_fixed_Q = (VarSize < rows_ct);
        static constexpr bool cols_fixed_Q = (VarSize < cols_ct);
        
        static constexpr Size_T M_ct = TransposedQ(op) ? cols_ct : rows_ct;
        static constexpr Size_T N_ct = TransposedQ(op) ? rows_ct : cols_ct;
        
        static constexpr bool M_fixed_Q = (VarSize < M_ct);
        static constexpr bool N_fixed_Q = (VarSize < N_ct);
        
        static constexpr bool M_small_Q = M_fixed_Q && (M_ct <= m_threshold);
        static constexpr bool N_small_Q = N_fixed_Q && (N_ct <= n_threshold);
        
    public:
        
        static constexpr Size_T m = M_small_Q ? M_ct : m_default;
        static constexpr Size_T n = N_small_Q ? N_ct : n_default;
        
    private:
        
        static constexpr Size_T rows_block_count_ct         { NotTransposedQ(op)       ? CeilDivide (rows_ct,m)   : CeilDivide (cols_ct,n)   };
        static constexpr Size_T cols_block_count_ct         { NotTransposedQ(op)       ? CeilDivide (cols_ct,n)   : CeilDivide (rows_ct,m)   };
        static constexpr Size_T rows_full_block_count_ct    { NotTransposedQ(op)       ? FloorDivide(rows_ct,m)   : FloorDivide(cols_ct,n)   };
        static constexpr Size_T cols_full_block_count_ct    { NotTransposedQ(op)       ? FloorDivide(cols_ct,n)   : FloorDivide(rows_ct,m)   };
        static constexpr Size_T rows_reg_ct                 { NotTransposedQ(op)       ? m * rows_block_count_ct  : n * cols_block_count_ct  };
        static constexpr Size_T cols_reg_ct                 { NotTransposedQ(op)       ? n * cols_block_count_ct  : m * rows_block_count_ct  };
        static constexpr Size_T rows_tail_ct                { rows_ct - rows_reg_ct };
        static constexpr Size_T cols_tail_ct                { cols_ct - cols_reg_ct };
        static constexpr Size_T m_tail_ct                   { NotTransposedQ(op)       ? rows_tail_ct             : cols_tail_ct             };
        static constexpr Size_T n_tail_ct                   { NotTransposedQ(op)       ? cols_tail_ct             : rows_tail_ct             };
    
        static constexpr Size_T M_block_count_ct            { NotTransposedQ(store_op) ? rows_block_count_ct      : cols_block_count_ct      };
        static constexpr Size_T N_block_count_ct            { NotTransposedQ(store_op) ? cols_block_count_ct      : rows_block_count_ct      };
        static constexpr Size_T M_full_block_count_ct       { NotTransposedQ(store_op) ? rows_full_block_count_ct : cols_full_block_count_ct };
        static constexpr Size_T N_full_block_count_ct       { NotTransposedQ(store_op) ? cols_full_block_count_ct : rows_full_block_count_ct };
        
    public:
        
        using Int = Size_T;
        
        using Block_T = Tiny::Matrix<m,n,Scal,Int>;
    
    private:
        
        // Sizes at runtime.

        const Size_T rows_rt;
        const Size_T cols_rt;
        
        const Size_T rows_block_count_rt;
        const Size_T cols_block_count_rt;
        
        const Size_T rows_full_block_count_rt;
        const Size_T cols_full_block_count_rt;
        
        const Size_T rows_reg_rt;
        const Size_T cols_reg_rt;
        
        const Size_T rows_tail_rt;
        const Size_T cols_tail_rt;
        
        const Size_T m_tail_rt;
        const Size_T n_tail_rt;
        
//        const Size_T M_rt;
//        const Size_T N_rt;
        
        const Size_T thread_count = 1;
        
        const Size_T M_block_count_rt;
        const Size_T N_block_count_rt;
        
        const Size_T M_full_block_count_rt;
        const Size_T N_full_block_count_rt;
        
//        const Size_T M_reg_rt;
//        const Size_T N_reg_rt;
//
//        const Size_T M_tail_rt;
//        const Size_T N_tail_rt;
        
    public:
        
        Tensor2<Block_T,Int> blocks;
        
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
        
        MatrixBlockMajor( const Size_T rows_rt_ = rows_ct, const Size_T cols_rt_ = cols_ct, const Size_T thread_count_ = 1 )
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
        ,   blocks                      ( M_block_count_rt, N_block_count_rt )
        {
            
        }
        
        ~MatrixBlockMajor() = default;
        
    public:
        
        
        mptr<Block_T> operator[]( const Size_T M_blk )
        {
            return blocks[M_blk];
        }
        
        cptr<Block_T> operator[]( const Size_T M_blk ) const
        {
            return blocks[M_blk];
        }
        
        mref<Block_T> operator()( const Size_T M_blk, const Size_T N_blk )
        {
            return blocks(M_blk,N_blk);
        }
        
        cref<Block_T> operator()( const Size_T M_blk, const Size_T N_blk ) const
        {
            return blocks(M_blk,N_blk);
        }

        Size_T RowCount() const
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
        
        Size_T ColCount() const
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
        
        Size_T M_FullBlockCount() const
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
        
        Size_T N_FullBlockCount() const
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
        
        Size_T M_BlockCount() const
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
        
        Size_T N_BlockCount() const
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

        Size_T RowTailCount() const
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

        Size_T ColTailCount() const
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
        
        
        Size_T BlockPosition( const Size_T M_blk, const Size_T N_blk, const Size_T ldA )
        {
            /// For the block with coordinates `{ M_blk, N_blk }` in `block`
            /// this computes linear index of the top-left entry of
            /// the corresponding matrix block i A-block.
            ///
            /// If `NotTransposedQ(op)`, then `A` is a matrix of
            /// size `rows_rt` x `cols_rt `, and it is supposed to be decomposed into
            /// blocks of size `m` x `n`.
            ///
            /// If `TransposedQ<op>`, then `A` is a matrix of
            /// size `cols_rt` x `rows_rt `, and it is supposed to be decomposed into
            /// blocks of size `n` x `m`.
            ///
            /// In any case, the leading dimension of `A` is `ldA`

            
            if constexpr ( NotTransposedQ(op) )
            {
                /// `A` is a matrix of size `rows_rt` x `cols_rt `,
                /// and it is supposed to be decomposed into blocks of size `m` x `n`.

                if constexpr ( NotTransposedQ(store_op) )
                {
                    /// The block `{ M_blk, N_blk }` corresponds to the `m` x `n` in `A`
                    /// with coordinates `{ M_blk, N_blk }`.
                    /// So the linear index of the top-left entry on this A-block is:
                    return ldA * m * M_blk + n * N_blk;
                }
                else if constexpr ( TransposedQ(store_op) )
                {
                    /// The block `{ M_blk, N_blk }` corresponds to the `m` x `n` in `A`
                    /// with coordinates `{ N_blk, M_blk }`.
                    /// So the linear index of the top-left entry on this A-block is:
                    return ldA * m * N_blk + n * M_blk;
                }
            }
            else if constexpr ( TransposedQ(op) )
            {
                /// If `TransposedQ<op>`, then `A` is a matrix of
                /// size `cols_rt` x `rows_rt`, and it is supposed to be decomposed into
                /// blocks of size `n` x `m`.
                
                if constexpr ( NotTransposedQ(store_op) )
                {
                    /// The block `{ M_blk, N_blk }` corresponds to the `n` x `m` in `A`
                    /// with coordinates `{ N_blk, M_blk }`.
                    /// So the linear index of the top-left entry on this A-block is:
                    return ldA * n * N_blk + m * M_blk;
                }
                else if constexpr ( TransposedQ(store_op) )
                {
                    /// The block `{ M_blk, N_blk }` corresponds to the `n` x `m` in `A`
                    /// with coordinates `{ M_blk, N_blk }`.
                    /// So the linear index of the top-left entry on this A-block is
                    return ldA * n * M_blk + m * N_blk;
                }
            }
        }
            
        template<typename A_T, typename A_I>
        void Read( cptr<A_T> A, const A_I ldA_ )
        {
            tic(ClassName()+"::Read<" + TypeName<A_T> + "," + TypeName<A_I> + ">");

            if( ldA_ <= 0 )
            {
                eprint(ClassName()+"::Read<" + TypeName<A_T> + "," + TypeName<A_I> + ">: leading dimensions <= 0 ");
                return;
            }
            
            const Size_T ldA = static_cast<Size_T>(ldA_);
            
            if constexpr ( NotTransposedQ(store_op) == NotTransposedQ(op) )
            {
                Read_N( A, ldA );
            }
            else
            {
                Read_T( A, ldA );
            }
            
            toc(ClassName()+"::Read<" + TypeName<A_T> + "," + TypeName<A_I> + ">");
        }
            
    private:

        template<typename A_T>
        void Read_N( cptr<A_T> A, const Size_T ldA )
        {
            //TODO: Handle boundary cases.

            Do<parQ>(
                [A,ldA,this]( const Size_T thread )
                {
                    constexpr Size_T M_step = NotTransposedQ(op) ? m : n;
                    constexpr Size_T N_step = NotTransposedQ(op) ? n : m;
                    
                    const Size_T M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
                    const Size_T M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

                    for( Size_T M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                    {
                        for( Size_T N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                        {
                            blocks[M_blk][N_blk].template Read<op>(
                                &A[ldA * M_step * M_blk + N_step * N_blk], ldA
                            );
                        }
                        
    //                    if( N_BlockCount() > N_FullBlockCount() )
    //                    {
    //                        const Size_T n_chopped = NotTransposedQ(op) ? RowTailCount() : ColTailCount();
    //
    //                        const Size_T N_blk = N_FullBlockCount();
    //
    //                        blocks[M_blk][N_blk].template Read<op>( m, n_chopped, &A[ldA * M_step * M_blk + N_step * N_blk], ldA );
    //                    }
                    }
                },
                thread_count, thread_count
            );
        }
            
        template<typename A_T>
        void Read_T( cptr<A_T> A, const Size_T ldA )
        {
            //TODO: Handle boundary cases.

            Do<parQ>(
                [A,ldA,this]( const Size_T thread )
                {
                    constexpr Size_T M_step = NotTransposedQ(op) ? m : n;
                    constexpr Size_T N_step = NotTransposedQ(op) ? n : m;
                    
                    const Size_T M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
                    const Size_T M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

                    for( Size_T N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                    {
                        for( Size_T M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                        {
                            blocks[M_blk][N_blk].template Read<op>(
                                &A[ldA * M_step * N_blk + N_step * M_blk], ldA
                            );
                        }
                    }
                },
                thread_count, thread_count
            );
        }
    
    public:
            
        template<
            typename alpha_T, typename beta_T, typename C_T, typename C_I
        >
        void Write(
                const alpha_T alpha, const beta_T beta,
                mptr<C_T> C, const C_I ldC_
        )
        {
            tic(ClassName()+"::Write<" + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + "," + TypeName<C_I> + ">");
            
            if( ldC_ <= 0 )
            {
                eprint(ClassName()+"::Write<" + "," + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + "," + TypeName<C_I> + ">: leading dimension <= 0 ");
                return;
            }
            
            const Size_T ldC = static_cast<Size_T>(ldC_);
            
            using F = Scalar::Flag;
            
            if( alpha == alpha_T(1) )
            {
                if( beta == beta_T(0) )
                {
                    Write_imp<F::Plus,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(1) )
                {
                    Write_imp<F::Plus,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(-1) )
                {
                    Write_imp<F::Plus,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_imp<F::Plus,F::Generic>( alpha, beta, C, ldC );
                }
            }
            else if( alpha == alpha_T(0) )
            {
                if( beta == beta_T(0) )
                {
                    Write_imp<F::Zero,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(1) )
                {
                    Write_imp<F::Zero,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(-1) )
                {
                    Write_imp<F::Zero,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_imp<F::Zero,F::Generic>( alpha, beta, C, ldC );
                }
            }
            else if( alpha == alpha_T(-1) )
            {
                if( beta == beta_T(0) )
                {
                    Write_imp<F::Minus,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(1) )
                {
                    Write_imp<F::Minus,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(-1) )
                {
                    Write_imp<F::Minus,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_imp<F::Minus,F::Generic>( alpha, beta, C, ldC );
                }
            }
            else
            {
                if( beta == beta_T(0) )
                {
                    Write_imp<F::Generic,F::Zero>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(1) )
                {
                    Write_imp<F::Generic,F::Plus>( alpha, beta, C, ldC );
                }
                else if( beta == beta_T(-1) )
                {
                    Write_imp<F::Generic,F::Minus>( alpha, beta, C, ldC );
                }
                else
                {
                    Write_imp<F::Generic,F::Generic>( alpha, beta, C, ldC );
                }
            }
            
            toc(ClassName()+"::Write<" + "," + TypeName<alpha_T> + "," + TypeName<beta_T> + "," + TypeName<C_T> + "," + TypeName<C_I> + ">");
        }
    private:
        
        template<
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename alpha_T, typename beta_T, typename C_T
        >
        void Write_imp(
                const alpha_T alpha, const beta_T beta,
                mptr<C_T> C, const Size_T ldC
        )
        {
            if constexpr ( NotTransposedQ(store_op) )
            {
                Write_imp_N<alpha_flag,beta_flag>( alpha, beta, C, ldC );
            }
            else if constexpr ( TransposedQ(store_op) )
            {
                Write_imp_T<alpha_flag,beta_flag>( alpha, beta, C, ldC );
            }
        }
        
        template<
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename alpha_T, typename beta_T, typename C_T
        >
        void Write_imp_N(
                const alpha_T alpha, const beta_T beta,
                mptr<C_T> C, const Size_T ldC
        )
        {
            //TODO: Handle boundary cases.
            
            ParallelDo(
                [=,this]( const Size_T thread )
                {
                    const Size_T M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
                    const Size_T M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

                    for( Size_T M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                    {
                        for( Size_T N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                        {
                            blocks[M_blk][N_blk].template Write<alpha_flag,beta_flag,op,Op::Id>(
                                alpha, beta, &C[ldC * m * M_blk + n * N_blk], ldC
                            );
                        }
                    }
                },
                thread_count
            );
        }
            
        template<
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename alpha_T, typename beta_T, typename C_T
        >
        void Write_imp_T(
                const alpha_T alpha, const beta_T beta,
                mptr<C_T> C, const Size_T ldC
        )
        {
            //TODO: Handle boundary cases.
            
            //TODO: Test this code branch!

            ParallelDo(
                [C,ldC,this]( const Size_T thread )
                {
                    constexpr Size_T M_step = NotTransposedQ(op) ? m : n;
                    constexpr Size_T N_step = NotTransposedQ(op) ? n : m;
                    
                    const Size_T M_blk_begin = JobPointer( M_FullBlockCount(), thread_count, thread     );
                    const Size_T M_blk_end   = JobPointer( M_FullBlockCount(), thread_count, thread + 1 );

                    for( Size_T N_blk = 0; N_blk < N_FullBlockCount(); ++N_blk )
                    {
                        for( Size_T M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
                        {
                            blocks[M_blk][N_blk].template Write<alpha_flag,beta_flag,op,Op::Id>(
                                &C[ldC * M_step * N_blk + N_step * M_blk], ldC
                            );
                        }
                    }
                },
                thread_count
            );
        }
            
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


