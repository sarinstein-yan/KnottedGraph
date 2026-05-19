Int BlockPosition( const Int M_blk, const Int N_blk, const Int ldA )
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
