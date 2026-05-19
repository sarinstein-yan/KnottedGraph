public:
    
template<typename ExtScal, IntQ ExtInt>
void FromTriples(
    const ExtInt  * const * const idx,            // list of lists of i-indices
    const ExtInt  * const * const jdx,            // list of lists of j-indices
    const ExtScal * const * const val,            // list of lists of nonzero values
    const LInt            * const entry_counts,   // list of lengths of the lists above
    const Int list_count,                         // number of lists
    const Int final_thread_count,                 // number of threads that the matrix shall use
    const bool compressQ = true,                  // whether to do additive assembly or not
    int symmetrizeQ = 0,
    int assemblerQ = false
)
{
    TOOLS_PTIMER(timer,ClassName()+"::FromTriples"
        + "<" + TypeName<ExtScal>
        + "," + TypeName<ExtInt>
        + ">("+ ToString(assemblerQ)
        + "," + ToString(symmetrizeQ)
        + ")"
    );
    
    Tensor1<LInt,Int> acc_entry_counts ( list_count + Int(1) );
    acc_entry_counts[0] = LInt(0);
    
    for( Int i = 0; i < list_count; ++i )
    {
        acc_entry_counts[i+1] = acc_entry_counts[i] + entry_counts[i];
    }
    
    LInt triple_count = acc_entry_counts.Last();
    
//    TOOLS_PDUMP(triple_count);
    
    if( triple_count <= LInt(0) )
    {
        SetThreadCount( final_thread_count );
        return;
    }
    
    Tensor2<LInt,Int> counters = AssemblyCounters<LInt,Int,parQ>(
        idx, jdx, entry_counts, list_count, m, symmetrizeQ
    );
    
    LInt nnz;
    
    if( list_count <= Int(0) )
    {
        eprint(ClassName()+"::FromTriples: list_count <= 0");
        nnz = LInt(0);
    }
    else
    {
        nnz = counters[list_count-Int(1)][m-Int(1)];
    }
    
//    TOOLS_PDUMP(nnz);
    
    if( nnz <= LInt(0) )
    {
        SetThreadCount( final_thread_count );
        return;
    }
    
    inner  = Tensor1<Int ,LInt>( nnz );
    values = Tensor1<Scal,LInt>( nnz );

//    TOOLS_PDUMP(outer.Size());
//    TOOLS_PDUMP(inner.Size());
//    TOOLS_PDUMP(values.Size());
    
    Tensor1<LInt,LInt> from;
    
    if ( assemblerQ )
    {
        from = Tensor1<LInt,LInt>( nnz );
    }
    
    mptr<LInt> A_o = outer.data();
    mptr< Int> A_i = inner.data();
    mptr<Scal> A_v = values.data();
    mptr<LInt> A_f = from.data();
    
    copy_buffer( counters.data(list_count-Int(1)), &A_o[1], m );

    //TOOLS_PDUMP(outer.Last());
    // The counters array tells each thread where to write.
    // Since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
    
    // TODO: The threads write quite chaotically to inner_ and value_. This might cause a lot of false sharing.
    
    // TODO: False sharing can be prevented by not distributing whole sublists of idx, jdx, val to the threads but by distributing the rows of the final matrix, instead. It's just a bit fiddly, though.
    
    // Writing reordered data.
    Do<parQ>(
        [
            assemblerQ,symmetrizeQ,A_i,A_v,A_f,
            &counters,&entry_counts,&acc_entry_counts,&idx,&jdx,&val
        ](
            const Int thread
        )
        {
            const LInt entry_count = entry_counts[thread];
            
            cptr<ExtInt>  t_idx = idx[thread];
            cptr<ExtInt>  t_jdx = jdx[thread];
            cptr<ExtScal> t_val = val[thread];
            
            mptr<LInt> c = counters.data(thread);
            
            LInt f = acc_entry_counts[thread+Int(1)];
            
            for( LInt k = entry_count; k --> LInt(0); )
            {
                const Int  i = static_cast<Int >(t_idx[k]);
                const Int  j = static_cast<Int >(t_jdx[k]);
                const Scal a = static_cast<Scal>(t_val[k]);
                
                {
                    const LInt pos = --c[i];
                    A_i[pos] = j;
                    A_v[pos] = a;
                    
                    if ( assemblerQ )
                    {
                        A_f[pos] = --f;
                    }
                }
                
                if ( symmetrizeQ != 0 )
                {
                    // Write the transposed matrix (diagonal excluded) in the same go in order to symmetrize the matrix. (Typical use case: Only the upper triangular part of a symmetric matrix is stored in idx, jdx, and val, but we need the full, symmetrized matrix.)
                    if( i != j )
                    {
                        const LInt pos = --c[j];
                        A_i[pos] = i;
                        A_v[pos] = a;
                        
                        if ( assemblerQ )
                        {
                            A_f[pos] = f;
                        }
                    }
                }
            }
        },
        list_count, list_count
    );
    
    
    // Now all j-indices and nonzero values lie in the correct row (as indexed by outer).

    // From here on, we may use as many threads as we want.
    SetThreadCount( final_thread_count );
    
    // We have to sort b_inner to be compatible with the CSR format.
    if( assemblerQ )
    {
        this->template SortInner_impl<true>( A_o, A_i, A_v, A_f );
    }
    else
    {
        this->template SortInner_impl<false>( A_o, A_i, A_v, A_f );
    }
    
    // Deal with duplicated {i,j}-pairs (additive assembly).
    if( compressQ )
    {
        if( assemblerQ )
        {
            Tensor1<LInt,LInt> C_outer;
            
            this->template Compress_impl<true,true>( outer, inner, values, C_outer );

            assembler = Assembler_T(
                std::move(C_outer),
                std::move(from),
                inner.Size(), triple_count, final_thread_count
            );
            
            assembler.SortInner();
        }
        else
        {
            Tensor1<LInt,LInt> C_outer;
            
            this->template Compress_impl<true,false>( outer, inner, values, C_outer );
        }
    }
    else
    {
        if( assemblerQ )
        {
            // TODO: In this case the `assembler` is a mere permutation matrix.
            // TODO: It would be more efficient to use the `Permutation` class instead.

            assembler = Assembler_T(
                iota<LInt,LInt>(inner.Size() + LInt(1)),
                std::move(from),
                inner.Size(), triple_count, final_thread_count
            );
        }
        
        // TODO: Build assembler
        proven_duplicate_freeQ = false;
    }
    
}
