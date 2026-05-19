public:

    const Tree_T & EliminationTree()
    {
        // TODO: How can this be parallelized?

        // TODO: read Kumar, Kumar, Basu - A parallel algorithm for elimination tree computation and symbolic factorization

        if( ! eTree_initializedQ )
        {
            TOOLS_PTIMER(timer,ClassName()+"::EliminationTree");

            // See Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
            // See Scott, Tuma - Algorithms for Sparse Linear Systems, Algorithm 4.2

            // I want to make it possible to use unsigned integer types for Int.
            // Hence, using -1 as "no_element" is not an option.
            // We have to use something else instead of 0 to mark empty places.
            
            
//            const Int no_element = n;
//
//            Tensor1<Int,Int> parents ( n, no_element );
//
//            // A vector for path compression.
//            Tensor1<Int,Int> a ( n, no_element );
//
//            cptr<LInt> A_diag  = A.Diag().data();
//            cptr<LInt> A_outer = A.Outer().data();
//            cptr< Int> A_inner = A.Inner().data();
//
//            for( Int k = 1; k < n; ++k )
//            {
//                // We need visit all i < k with A_ik != 0.
//                const LInt l_begin = A_outer[k];
//                const LInt l_end   = A_diag [k];
//
//                for( LInt l = l_begin; l < l_end; ++l )
//                {
//                    Int i = A_inner[l];
//
//                    while( ( i != no_element ) && ( i < k ) )
//                    {
//                        Int j = a[i];
//
//                        a[i] = k;
//
//                        if( j == no_element )
//                        {
//                            parents[i] = k;
//                        }
//                        i = j;
//                    }
//                }
//            }
            
            Tensor1<Int,Int> parents ( n );
            
            Sparse::EliminationTreeParents(
                n, A.Outer().data(), A.Inner().data(), parents.data()
            );

            eTree = Tree_T( std::move(parents), thread_count );

            eTree_initializedQ = true;
        }

        return eTree;
    }
