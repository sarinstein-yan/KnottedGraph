public:
    
    void SymbolicFactorization()
    {
        if( !SN_initializedQ )
        {
            TOOLS_PTIMER(timer,ClassName()+"::SymbolicFactorization");
            
            (void)PostOrdering();

            switch ( SN_strategy )
            {
                case SupernodeStrategy_T::Maximal:
                {
                    FindMaximalSupernodes();
                    break;
                }
                case SupernodeStrategy_T::Fundamental:
                {
                    FindFundamentalSupernodes();
                    break;
                }
                case SupernodeStrategy_T::Amalgamated:
                {
                    FindAmalgamatedSupernodes();
                    break;
                }
                default:
                {
                    FindMaximalSupernodes();
                    break;
                }
            }
            
            // TODO: Maybe resizing these arrays is unnecessary. But it is quite fast, anyway.
            SN_rp   .template Resize<true>( SN_count + 1 );
            SN_outer.template Resize<true>( SN_count + 1 );
            
            SN_tri_ptr = Tensor1<LInt,Int>( SN_count + 1 );
            SN_tri_ptr[0] = 0;
            
            SN_rec_ptr = Tensor1<LInt,Int>( SN_count + 1 );
            SN_rec_ptr[0] = 0;
            
            max_n_0 = 0;
            max_n_1 = 0;
            
            for( Int k = 0; k < SN_count; ++k )
            {
                // Warning: Taking differences of potentially signed numbers.
                // Should not be of concern because negative numbers appear here only if something went wrong upstream.
                const Int n_0 =               SN_rp   [k+1] - SN_rp   [k];
                const Int n_1 = int_cast<Int>(SN_outer[k+1] - SN_outer[k]);

                max_n_0 = Max( max_n_0, n_0 );
                max_n_1 = Max( max_n_1, n_1 );

                SN_tri_ptr[k+1] = SN_tri_ptr[k] + n_0 * n_0;
                SN_rec_ptr[k+1] = SN_rec_ptr[k] + n_0 * n_1;
            }
            
            CreateAssemblyTree();
            
            SN_initializedQ = true;
        }
    }


protected:

    void FindAmalgamatedSupernodes()
    {
        // Compute supernodal symbolic factorization with so-called _fundamental supernodes_.
        // See Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations,
        // https://www.osti.gov/servlets/purl/6756314
        
        // We avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

        TOOLS_PTIMER(timer,ClassName()+"::FindAmalgamatedSupernodes");
        
        // temporary arrays
    //            Tensor1<Int,Int> prev_row(n);// Array to aggregate column indices of previous row of U.
        Tensor1<Int,Int> curr_row(n);// Array to aggregate column indices of current  row of U.
        Tensor1<Int,Int> scratch (n);// Some scratch space for UniteSortedBuffers.
        
        Int curr_n_0 = 0; // Holds the number of rows in current supernode.
        Int curr_n_1 = 0; // Holds the number of column indices in current supernode.
        Int curr_i_0 = 0; // Holds the index  of first row in current supernode.

        Int prev_n_0 = 0; // Holds the number of rows in previous supernode.
        Int prev_n_1 = 0; // Holds the number of column indices in previous supernode.
    //            Int prev_i_0 = 0; // Holds the index of first row in previous supernode

        
        // TODO: Fix this to work with unsinged integers.
        Tensor1<Int,Int> prev_col_nz(n,Int(-1));

        // i-th row of U belongs to supernode row_to_SN[i].
        row_to_SN   = Tensor1< Int,Int> (n);
        
        // Holds the current number of supernodes.
        SN_count    = 0;
        // Pointers from supernodes to their starting rows.
        SN_rp       = Tensor1< Int,Int> (n+Int(2));
        
        // Pointers from supernodes to their starting position in SN_inner.
        SN_outer    = Tensor1<LInt,Int> (n+Int(2));
        SN_outer[0] = 0;
        
        // To be filled with the column indices of super nodes.
        // Will later be moved to SN_inner.
        Aggregator<Int,LInt> SN_inner_agg ( 2 * A.NonzeroCount(), thread_count );
        
        // Start first supernode.
        SN_rp[0]     = 0;
        row_to_SN[0] = 0;
        SN_count     = 0;
        
        // TODO: Should be parallelizable by processing subtrees of elimination tree in parallel.
        // TODO: Even better: Build aTree first (we need only to know the fundamental rows for that). Then collect SN_inner by traversing aTree in parellel.
        // TODO: --> Can we precompute somehow the size of SN_inner_agg? That would greatly help to reduce copy ops and to schedule its generation.
        
        for( Int i = 1; i < n + Int(1); ++i ) // Traverse rows.
        {
            if( FundamentalQ( i, prev_col_nz ) )
            {
                // i is going to be the first node of the newly created fundamental supernode.
                // However, we do not now at the moment how long the supernode is going to be.
                
                // Instead building the new supernode, we first have to finish the current supernode.
                
                curr_n_0 = i - curr_i_0;
                
                // The nonzero pattern of upper(A) belongs definitely to the pattern of U.
                // We have to find all nonzero columns j of row curr_i_0 of A such that j > i-1,
                // because that will be the ones belonging to the rectangular part.

                // We know that A.Inner(A.Diag(curr_i_0)) == curr_i_0 < i.
                // Hence, we can start the search here:
                {
                    LInt k = A.Diag(curr_i_0) + 1;
                    
                    const LInt k_end = A.Outer(curr_i_0+1);
                    
                    while( (A.Inner(k) < i) && (k < k_end) ) { ++k; }
                    
                    curr_n_1 = int_cast<Int>(k_end - k);
                    
                    copy_buffer( &A.Inner(k), curr_row.data(), curr_n_1 );
                }
                
                // Next, we have to merge the column indices of the children of i_0 into row.
                const Int l_begin = eTree.ChildPointer(curr_i_0  );
                const Int l_end   = eTree.ChildPointer(curr_i_0+1);
                
                // Traverse all children of curr_i_0 in the eTree. Most of the time it's zero, one or two children. Seldomly it's more.
                for( Int l = l_begin; l < l_end; ++l )
                {
                    const Int j = eTree.ChildIndex(l);
                    // We have to merge the column indices of child j that are greater than i into U_row.
                    // This is the supernode where we find the j-th row of U.
                    const Int k = row_to_SN[j];
                    
                    if( k == SN_count )
                    {
                        // No need to unite anything.
                        continue;
                    }
                    
                    // Notice that because of j < i, we only have to consider the reactangular part of this supernode.
                    
                          LInt a = SN_outer[k  ];
                    const LInt b = SN_outer[k+1];
                    
                    // Only consider column indices of j-th row of U that are greater than last row i-1 in current supernode.
                    while( (SN_inner_agg[a] < i) && (a < b) ) { ++a; }
                    
                    if( a < b )
                    {
                        curr_n_1 = UniteSortedBuffers(
                            curr_row.data(),  curr_n_1,
                            &SN_inner_agg[a], int_cast<Int>(b - a),
                            scratch.data()
                        );
                        swap( curr_row, scratch );
                    }
                }

                // TODO: Do some meaningful check here.
                bool amalgamateQ = (SN_count > Int(0)) && (curr_n_0 + prev_n_0 <= amalgamation_threshold);
                
                if( amalgamateQ )
                {
                    // We need to merge previous and current column indices.
                    
                    // Discard previous supernode's column indices that are < i.
                          LInt a = SN_outer[SN_count-Int(1)];
                    const LInt b = SN_outer[SN_count       ];

                    while( (SN_inner_agg[a] < i) && (a < b) ) { ++a; }
                    
                    if( a < b )
                    {
                        curr_n_1 = UniteSortedBuffers(
                            curr_row.data(),  curr_n_1,
                            &SN_inner_agg[a], int_cast<Int>(b - a),
                            scratch.data()
                        );
                        swap( curr_row, scratch );
                    }
                    
                    
                    // Remove current supernode.
                    --SN_count;
                    

                    
                    // Remove the column indices for previous supernode.
                    SN_inner_agg.Pop( prev_n_1 );

                    // Tell all rows of current supernode that they belong to the previous node instead.
                    for( Int k = i - curr_n_0; k < i; ++k )
                    {
                        row_to_SN[k] = SN_count;
                    }

                    curr_n_0 = prev_n_0 + curr_n_0;
                }
                
                // Now curr_row is ready to be pushed into SN_inner.
                SN_inner_agg.Push( curr_row.data(), curr_n_1 );

                // Start new supernode.
                ++SN_count;
                
                SN_outer[SN_count] = SN_inner_agg.Size();
                SN_rp   [SN_count] = curr_i_0 = i;
                
                prev_n_0 = curr_n_0;
                prev_n_1 = curr_n_1;
            }
            else
            {
                // Continue supernode.
            }
            
            // Remember where to find i-th row.
            if( i < n )
            {
                row_to_SN[i] = SN_count;
            }
            
        } // for( Int i = 0; i < n+1; ++i )
        
        SN_inner = std::move(SN_inner_agg.Disband());
    }

    void FindMaximalSupernodes()
    {
        // Compute supernodal symbolic factorization with so-called _fundamental supernodes_.
        // See Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations,
        // https://www.osti.gov/servlets/purl/6756314
        
        // We avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

        TOOLS_PTIMER(timer,ClassName()+"::FindMaximalSupernodes");
        
        // temporary arrays
//            Tensor1<Int,Int> prev_row(n);// Array to aggregate column indices of previous row of U.
        Tensor1<Int,Int> curr_row(n);// Array to aggregate column indices of current  row of U.
        Tensor1<Int,Int> scratch (n);// Some scratch space for UniteSortedBuffers.
        
        Int curr_n_0 = 0; // Holds the number of rows in current supernode.
        Int curr_n_1 = 0; // Holds the number of column indices in current supernode.
        Int curr_i_0 = 0; // Holds the index  of first row in current supernode.

        Int prev_n_0 = 0; // Holds the number of rows in previous supernode.
        Int prev_n_1 = 0; // Holds the number of column indices in previous supernode.
//            Int prev_i_0 = 0; // Holds the index of first row in previous supernode

        
        // TODO: Fix this to work with unsinged integers.
        Tensor1<Int,Int> prev_col_nz(n,-1);

        // i-th row of U belongs to supernode row_to_SN[i].
        row_to_SN   = Tensor1< Int,Int> (n);
        
        // Holds the current number of supernodes.
        SN_count    = 0;
        // Pointers from supernodes to their starting rows.
        SN_rp       = Tensor1< Int,Int> (n + Int(2));
        
        // Pointers from supernodes to their starting position in SN_inner.
        SN_outer    = Tensor1<LInt,Int> (n + Int(2));
        SN_outer[0] = 0;
        
        // To be filled with the column indices of super nodes.
        // Will later be moved to SN_inner.
        Aggregator<Int,LInt> SN_inner_agg ( 2 * A.NonzeroCount(), thread_count );
        
        // Start first supernode.
        SN_rp[0]     = 0;
        row_to_SN[0] = 0;
        SN_count     = 0;
        
        // TODO: Should be parallelizable by processing subtrees of elimination tree in parallel.
        // TODO: Even better: Build aTree first (we need only to know the fundamental rows for that). Then collect SN_inner by traversing aTree in parellel.
        // TODO: --> Can we precompute somehow the size of SN_inner_agg? That would greatly help to reduce copy ops and to schedule its generation.
        
        for( Int i = 1; i < n+1; ++i ) // Traverse rows.
        {
            if( FundamentalQ( i, prev_col_nz ) )
            {
                // i is going to be the first node of the newly created fundamental supernode.
                // However, we do not now at the moment how long the supernode is going to be.
                
                // Instead building the new supernode, we first have to finish the current supernode.
                
                curr_n_0 = i - curr_i_0;
                
                // The nonzero pattern of upper(A) belongs definitely to the pattern of U.
                // We have to find all nonzero columns j of row curr_i_0 of A such that j > i-1,
                // because that will be the ones belonging to the rectangular part.

                // We know that A.Inner(A.Diag(curr_i_0)) == curr_i_0 < i.
                // Hence, we can start the search here:
                {
                    LInt k = A.Diag(curr_i_0) + 1;
                    
                    const LInt k_end = A.Outer(curr_i_0+1);
                    
                    while( (A.Inner(k) < i) && (k < k_end) ) { ++k; }
                    
                    curr_n_1 = int_cast<Int>(k_end - k);
                    
                    copy_buffer( &A.Inner(k), curr_row.data(), int_cast<Size_T>(curr_n_1) );
                }
                
                // Next, we have to merge the column indices of the children of i_0 into row.
                const Int l_begin = eTree.ChildPointer(curr_i_0  );
                const Int l_end   = eTree.ChildPointer(curr_i_0+1);
                
                // Traverse all children of curr_i_0 in the eTree. Most of the time it's zero, one or two children. Seldomly it's more.
                for( Int l = l_begin; l < l_end; ++l )
                {
                    const Int j = eTree.ChildIndex(l);
                    // We have to merge the column indices of child j that are greater than i into U_row.
                    // This is the supernode where we find the j-th row of U.
                    const Int k = row_to_SN[j];
                    
                    if( k == SN_count )
                    {
                        // No need to unite anything.
                        continue;
                    }
                    
                    // Notice that because of j < i, we only have to consider the reactangular part of this supernode.
                    
                          LInt a = SN_outer[k       ];
                    const LInt b = SN_outer[k+Int(1)];
                    
                    // Only consider column indices of j-th row of U that are greater than last row i-1 in current supernode.
                    while( (SN_inner_agg[a] < i) && (a < b) ) { ++a; }
                    
                    if( a < b )
                    {
                        curr_n_1 = UniteSortedBuffers(
                            curr_row.data(),  curr_n_1,
                            &SN_inner_agg[a], int_cast<Int>(b - a),
                            scratch.data()
                        );
                        swap( curr_row, scratch );
                    }
                }

                if( SN_count > Int(0) )
                {
                    // Check whether previous row has same pattern.
                    
                    // a and b can be computed from SN_inner_agg.Size() and prev_n_1.

                    // TODO: Use prev_n_0 for this.
                    // Discard previous supernode's column indices that are < i.
                          LInt a = SN_outer[SN_count - Int(1)];
                    const LInt b = SN_outer[SN_count         ];
                    
                    while( (SN_inner_agg[a] < i) && (a < b) ) { ++a; }

                    if( (curr_n_1 > Int(0)) && (b - a  == static_cast<LInt>(curr_n_1)) )
                    {
                        bool mergeQ = true;
                        
                        for( Int c = 0; c < curr_n_1; ++ c )
                        {
                            mergeQ = mergeQ && SN_inner_agg[a+c] == curr_row[c];
                        }
                        
                        if( mergeQ )
                        {
//                                logprint("Merging supernodes " + ToString(SN_count-1) + " and " + ToString(SN_count) + "." );

                            // Remove current supernode.
                            --SN_count;
                            
                            // Remove the column indices for previous supernode.
                            SN_inner_agg.Pop( prev_n_1 );

                            // Tell all rows of current supernode that they belong to the previous node instead.
                            for( Int k = i - curr_n_0; k < i; ++k )
                            {
                                row_to_SN[k] = SN_count;
                            }

                            curr_n_0 = prev_n_0 + curr_n_0;
                        }
                    }
                }
                
                // Now curr_row is ready to be pushed into SN_inner.
                SN_inner_agg.Push( curr_row.data(), curr_n_1 );

                // Start new supernode.
                ++SN_count;
                
                SN_outer[SN_count] = SN_inner_agg.Size();
                SN_rp   [SN_count] = curr_i_0 = i;
                
                prev_n_0 = curr_n_0;
                prev_n_1 = curr_n_1;
            }
            else
            {
                // Continue supernode.
            }
            
            // Remember where to find i-th row.
            if( i < n )
            {
                row_to_SN[i] = SN_count;
            }
            
        } // for( Int i = 0; i < n+1; ++i )
        
        SN_inner = std::move(SN_inner_agg.Disband());
    }

    void FindFundamentalSupernodes()
    {
        TOOLS_PTIMER(timer,ClassName()+"FindFundamentalSupernodes");
        // Compute supernodal symbolic factorization with so-called _fundamental supernodes_.
        // See Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations,
        // https://www.osti.gov/servlets/purl/6756314
        
        // We avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

        // temporary arrays

        Tensor1<Int,Int> curr_row(n);// Array to aggregate column indices of current  row of U.
        Tensor1<Int,Int> scratch (n);// Some scratch space for UniteSortedBuffers.
        
        Int curr_n_0 = 0; // Holds the number of rows in current supernode.
        Int curr_n_1 = 0; // Holds the number of column indices in current supernode.
        Int curr_i_0 = 0; // Holds the index  of first row in current supernode.

        Int prev_n_0 = 0; // Holds the number of rows in previous supernode.
        Int prev_n_1 = 0; // Holds the number of column indices in previous supernode.
        
        // TODO: Fix this to work with unsinged integers.
        Tensor1<Int,Int> prev_col_nz(n,Int(-1));

        // i-th row of U belongs to supernode row_to_SN[i].
        row_to_SN   = Tensor1< Int,Int> (n);
        
        // Holds the current number of supernodes.
        SN_count    = 0;
        // Pointers from supernodes to their starting rows.
        SN_rp       = Tensor1< Int,Int> (n + Int(2));
        
        // Pointers from supernodes to their starting position in SN_inner.
        SN_outer    = Tensor1<LInt,Int> (n + Int(2));
        SN_outer[0] = 0;
        
        // To be filled with the column indices of super nodes.
        // Will later be moved to SN_inner.
        Aggregator<Int,LInt> SN_inner_agg ( LInt(2) * A.NonzeroCount(), thread_count );
        
        // Start first supernode.
        SN_rp[0]     = 0;
        row_to_SN[0] = 0;
        SN_count     = 0;
        
        // TODO: Should be parallelizable by processing subtrees of elimination tree in parallel.
        // TODO: Even better: Build aTree first (we need only to know the fundamental rows for that). Then collect SN_inner by traversing aTree in parellel.
        // TODO: --> Can we precompute somehow the size of SN_inner_agg? That would greatly help to reduce copy ops and to schedule its generation.
        
        for( Int i = 1; i < n+1; ++i ) // Traverse rows.
        {
            if( FundamentalQ( i, prev_col_nz ) )
            {
                // i is going to be the first node of the newly created fundamental supernode.
                // However, we do not now at the moment how long the supernode is going to be.
                
                // Instead building the new supernode, we first have to finish the current supernode.
                
                curr_n_0 = i - curr_i_0;
                
                // The nonzero pattern of upper(A) belongs definitely to the pattern of U.
                // We have to find all nonzero columns j of row curr_i_0 of A such that j > i-1,
                // because that will be the ones belonging to the rectangular part.

                // We know that A.Inner(A.Diag(curr_i_0)) == curr_i_0 < i.
                // Hence, we can start the search here:
                {
                    LInt k = A.Diag(curr_i_0) + 1;
                    
                    const LInt k_end = A.Outer(curr_i_0+1);
                    
                    while( (A.Inner(k) < i) && (k < k_end) ) { ++k; }
                    
                    curr_n_1 = int_cast<Int>(k_end - k);
                    
                    copy_buffer( &A.Inner(k), curr_row.data(), curr_n_1 );
                }
                
                // Next, we have to merge the column indices of the children of i_0 into row.
                const Int l_begin = eTree.ChildPointer(curr_i_0  );
                const Int l_end   = eTree.ChildPointer(curr_i_0+1);
                
                // Traverse all children of curr_i_0 in the eTree. Most of the time it's zero, one or two children. Seldomly it's more.
                for( Int l = l_begin; l < l_end; ++l )
                {
                    const Int j = eTree.ChildIndex(l);
                    // We have to merge the column indices of child j that are greater than i into U_row.
                    // This is the supernode where we find the j-th row of U.
                    const Int k = row_to_SN[j];
                    
                    if( k == SN_count )
                    {
                        // No need to unit  anything.
                        continue;
                    }
                    
                    // Notice that because of j < i, we only have to consider the reactangular part of this supernode.
                    
                          LInt a = SN_outer[k  ];
                    const LInt b = SN_outer[k+1];
                    
                    // Only consider column indices of j-th row of U that are greater than last row i-1 in current supernode.
                    while( (SN_inner_agg[a] < i) && (a < b) ) { ++a; }
                    
                    if( a < b )
                    {
                        curr_n_1 = UniteSortedBuffers(
                            curr_row.data(),  curr_n_1,
                            &SN_inner_agg[a], int_cast<Int>(b - a),
                            scratch.data()
                        );
                        swap( curr_row, scratch );
                    }
                }

                // Now curr_row is ready to be pushed into SN_inner.
                SN_inner_agg.Push( curr_row.data(), curr_n_1 );

                // Start new supernode.
                ++SN_count;
                
                SN_outer[SN_count] = SN_inner_agg.Size();
                SN_rp   [SN_count] = curr_i_0 = i;
                
                prev_n_0 = curr_n_0;
                prev_n_1 = curr_n_1;
            }
            else
            {
                // Continue supernode.
            }
            
            // Remember where to find i-th row.
            if( i < n )
            {
                row_to_SN[i] = SN_count;
            }
            
        } // for( Int i = 0; i < n+1; ++i )
        
        SN_inner = std::move(SN_inner_agg.Disband());
    }
    
    bool FundamentalQ( const Int i, mref<Tensor1<Int,Int>> prev_col_nz )
    {
        // Using Theorem 2.3 and Corollary 3.2 in
        //
        //     Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations,
        //     https://www.osti.gov/servlets/purl/6756314
        //
        // to determine whether a new fundamental supernode starts at node u.
        
        bool is_fundamental = ( i == n ); // We make virtual root vertex fundamental, so that the main loop finishes off the last nonvirtual supernode correctly.
        
        is_fundamental = is_fundamental || ( eTree.ChildCount(i) > Int(1));
        
        if( !is_fundamental )
        {
            const Int threshold = i - eTree.DescendantCount(i) + 1;

            const LInt k_begin = A.Diag(i)+1; // exclude diagonal entry
            const LInt k_end   = A.Outer(i+1);
            
            for( LInt k = k_begin; k < k_end; ++k )
            {
                const Int j = A.Inner(k);
                const Int l = prev_col_nz[j];
                
                prev_col_nz[j] = i;
                
                is_fundamental = is_fundamental || ( l < threshold );
            }
        }
        
        return is_fundamental;
    }


    bool TestRowToSupernode()
    {
        TOOLS_PTIMER(timer,ClassName()+"TestRowToSupernode");
        
        bool succeededQ = true;
        
        for( Int s = 0; s < SN_count; ++s )
        {
            const Int i_begin = SN_rp[s    ];
            const Int i_end   = SN_rp[s + 1];
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                if( row_to_SN[i] != s)
                {
                    succeededQ = false;
                    
                    eprint(std::string("row_to_SN[")+ToString(i)+"] != " + ToString(s));
                }
            }
        }
        
        return succeededQ;
    }
