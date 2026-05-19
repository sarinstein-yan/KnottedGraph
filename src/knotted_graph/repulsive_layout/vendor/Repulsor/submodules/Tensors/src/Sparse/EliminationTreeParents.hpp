#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<IntQ Int, IntQ Int2, IntQ LInt>
        void EliminationTreeParents(
            const Int  n,
            cptr<LInt> A_outer,
            cptr<Int2> A_inner,
            mptr< Int> parents
        )
        {
            // We assume that A_outer, A_inner defines a CSR pattern.
            // In particular, we assume that
            //
            //    [ A_inner[A_outer[k]], ..., A_inner[A_outer[k+1]] [
            //
            // is ordered.
            
            TOOLS_PTIMER(timer,std::string("Sparse::EliminationTreeParents")
                 + "<" + TypeName<Int>
                 + "," + TypeName<Int2>
                 + "," + TypeName<LInt>
                 + ">");

            // I want to make it possible to use unsigned integer types for Int.
            // Hence, using -1 as "no_element" is not an option.
            // We have to use something else instead of 0 to mark empty places.
            const Int no_element = n;

            fill_buffer( parents, no_element, n );

            // A vector for path compression.
            Tensor1<Int,Int> a ( n, no_element );

            for( Int k = 1; k < n; ++k )
            {
                // We need visit all i < k with A_ik != 0.
                const LInt l_begin = A_outer[k    ];
                const LInt l_end   = A_outer[k + 1];

                for( LInt l = l_begin; l < l_end; ++l )
                {
                    Int i = int_cast<Int>(A_inner[l]);

                    while( ( i != no_element ) && ( i < k ) )
                    {
                        Int j = a[i];

                        a[i] = k;

                        if( j == no_element )
                        {
                            parents[i] = k;
                        }
                        
                        i = j;
                    }
                }
            }
        }
        
        
        template<IntQ Int, IntQ Int2, IntQ LInt, bool checkQ = true>
        bool PermutedEliminationTreeParents(
            const Int  n,
            cptr<LInt> A_outer,
            cptr<Int2> A_inner,
            cptr< Int> perm,           // permutation
            cptr< Int> perm_inv,       // inverse permutation
            mptr< Int> parents
        )
        {
            std::string tag = std::string("Sparse::PermutedEliminationTreeParents")
                + "<" + TypeName<Int>
                + "," + TypeName<Int2>
                + "," + TypeName<LInt>
                + "," + ToString(checkQ)
                + ">";
            
            // We want to compute the elimination tree of B = A[p,p].
            TOOLS_PTIMER(timer,tag);

            // I want to make it possible to use unsigned integer types for Int.
            // Hence, using -1 as "no_element" is not an option.
            // We have to use something else instead of 0 to mark empty places.
            const Int no_element = n;

            fill_buffer( parents, no_element, n );

            // A vector for path compression.
            Tensor1<Int,Int> a ( n, no_element );
            
            // A buffer for working with the column indices.
            Tensor1<Int,LInt> idx ( n );

            for( Int k = 1; k < n; ++k )
            {
                const Int p_k = perm[k];
                
                if constexpr ( checkQ )
                {
                    if( p_k < Int(0) )
                    {
                        eprint( tag + ": perm[" + ToString(k) + "] < 0.");
                        return false;
                    }
                    
                    if( p_k > n )
                    {
                        eprint( tag + ": perm[" + ToString(k) + "] > " + ToString(n) + ".");
                        return false;
                    }
                }
                
                // We need visit all i < k with B_ik = B_ki != 0.
                
                // We first fetch the k-th row of B.
                
                const LInt l_begin = A_outer[p_k    ];
                const LInt l_end   = A_outer[p_k + 1];
                
                const LInt row_length = l_end - l_begin;
                
                for( LInt l = l_begin; l < l_end; ++l )
                {
                    const Int j = int_cast<Int>(A_inner[l]);
                    
                    const Int q_j = perm_inv[ j ];
                    
                    if constexpr ( checkQ )
                    {
                        if( q_j < Int(0) )
                        {
                            eprint( tag + ": perm_inv[" + ToString(j) + "] < 0.");
                            return false;
                        }
                        
                        if( q_j > n )
                        {
                            eprint(tag  + ": perm_inv[" + ToString(j) + "] > " + ToString(n) + ".");
                            return false;
                        }
                    }
                    
                    idx[l-l_begin] = q_j;
                }
                
                Sort( &idx[0], &idx[row_length] );

                for( LInt l = 0; l < row_length; ++l )
                {
                    Int i = idx[l];

                    while( ( i != no_element ) && ( i < k ) )
                    {
                        Int j = a[i];

                        a[i] = k;

                        if( j == no_element )
                        {
                            parents[i] = k;
                        }
                        
                        i = j;
                    }
                }
            }
            return true;
        }
        
        
        template<IntQ Int>
        bool PostOrderedQ(
            const Int n,
            cptr<Int> parents,
            mptr<Int> descendant_counts
        )
        {
            // parents is assumed to be an array of n integers
            // in the range [0,...,n] _with n included.
            // node n acts as root.
            
            // descendant_counts has to be an array of n+1(!) integers,
            // so that every node, including the root, has a valid number of descendants.

            TOOLS_PTIMER(timer,std::string("Sparse::PostOrderedQ")
                + "<" + TypeName<Int>
                + ">");
            
            fill_buffer( descendant_counts, Int(1), n+1 );
            
            // Accumulate the descendant counts.
            for( Int i = 0; i < n; ++i )
            {
                const Int p_i = parents[i];
                
                if( i >= p_i ) { return false; }
                
                descendant_counts[p_i] += descendant_counts[i];
            }
            
            // Finally check for postordering.
            for( Int i = 0; i < n; ++i )
            {
                const Int p_i = parents[i];
                
                // TODO: Is this test good enough to indeed guarantee that the tree is really postordered?
                // Checking that i - descendant_counts[i] >= p_i - descendant_counts[p_i].
                
                if( i + descendant_counts[p_i] < p_i + descendant_counts[i]  )
                {
                    return false;
                }
            }
            return true;
        }
        
        

    } // namespace Sparse
    
} // namespace Tensors
