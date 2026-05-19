protected:

template<bool valuesQ, bool assemblerQ, typename T>
void Compress_impl(
    mref<Tensor1<LInt, Int>> A_outer,
    mref<Tensor1< Int,LInt>> A_inner,
    mref<Tensor1<   T,LInt>> A_values,
    mref<Tensor1<LInt,LInt>> C_outer
)  const
{
    TOOLS_PTIMER(timer,ClassName()+"::Compress_impl"
        + "<" + BoolString(valuesQ)
        + "," + BoolString(assemblerQ)
        + "," + TypeName<T> + ">"
    );
    
//    TOOLS_PDUMP(A_outer.Size());
//    TOOLS_PDUMP(A_inner.Size());
//    TOOLS_PDUMP(A_values.Size());
    
    if( proven_duplicate_freeQ ) { return; }
    
    if( !this->WellFormedQ() ) { return; }
    
    RequireJobPtr();
    SortInner();

    Tensor1<LInt,Int> B_outer (A_outer.Size());
    B_outer[0] = LInt(0);

    Tensor1<LInt,LInt> contraction_counts;
    
    if constexpr ( assemblerQ )
    {
        contraction_counts = Tensor1<LInt,LInt>(A_inner.Size() + LInt(1));
    }

    {
        cptr<LInt> A_o = A_outer.data();
        mptr< Int> A_i = A_inner.data();
        mptr<   T> A_v = A_values.data();
        
        mptr<LInt> B_o = B_outer.data();
        mptr<LInt> c_c = contraction_counts.data();
        
        
        Do<parQ>(
            [A_o,A_i,A_v,B_o,c_c,this]( const Int thread )
            {
                if constexpr( !valuesQ )
                {
                    (void)A_v;
                }
                if constexpr( !assemblerQ )
                {
                    (void)c_c;
                }
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                // To where we write.
                LInt k_new        = A_o[i_begin];
                LInt next_k_begin = A_o[i_begin];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt k_begin = next_k_begin;
                    const LInt k_end   = A_o[i+1];
                    
                    // Memorize the next entry in outer because outer will be overwritten
                    next_k_begin = k_end;
                    
                    LInt row_nonzero_counter = 0;
                    
                    // From where we read.
                    LInt k = k_begin;
                    T a;
                    
                    LInt c_counter = 0;
                    
                    while( k < k_end )
                    {
                        Int  j = A_i[k];

                        if constexpr ( valuesQ )
                        {
                            a = A_v[k];
                        }
                        if constexpr ( assemblerQ )
                        {
                            c_counter = LInt(1);
                        }
                        ++k;
                        
                        while( (k < k_end) && (j == A_i[k]) )
                        {
                            if constexpr ( valuesQ )
                            {
                                a += A_v[k];
                            }
                            if constexpr ( assemblerQ )
                            {
                                ++c_counter;
                            }
                            ++k;
                        }
                        
                        A_i[k_new] = j;
                        
                        if constexpr ( valuesQ )
                        {
                            A_v[k_new] = a;
                        }
                        
                        if constexpr ( assemblerQ )
                        {
                            c_c[k_new] = c_counter;
                        }
                        
                        ++k_new;
                        ++row_nonzero_counter;
                    }
                    
                    B_o[i+1] = row_nonzero_counter;
                }
            },
            job_ptr.ThreadCount(), job_ptr.ThreadCount()
        );
    }

    // TODO: Test what works better.
    B_outer.template Accumulate<parQ>( thread_count );
//    B_outer.Accumulate(); // I think I prefer the sequential code here.
    
    // Now B_outer is the new array of outer indices.
    
    const LInt nnz = B_outer[m];
    
    // Now we create new arrays for B_inner and B_values.
    // Then we copy A_inner and A_values to it, eliminating the gaps in between.
    
    Tensor1<Int,LInt> B_inner  (nnz);
    Tensor1<  T,LInt> B_values ( valuesQ ? nnz : LInt(0) );

    C_outer = Tensor1<LInt,LInt>(nnz + LInt(1));
    C_outer[0] = LInt(0);
    
//    TOOLS_PDUMP(C_outer.Size());
    
    {
        cptr<LInt> A_o = A_outer.data();
        cptr< Int> A_i = A_inner.data();
        cptr<   T> A_v = A_values.data();
        
        cptr<LInt> B_o = B_outer.data();
        mptr< Int> B_i = B_inner.data();
        mptr<   T> B_v = B_values.data();
        
        mptr<LInt> c_c = contraction_counts.data();
        mptr<LInt> C_o = C_outer.data();
        
        //TODO: Parallelization might be a bad idea here.
        Do<parQ>(
            [A_o,A_i,A_v,B_o,B_i,B_v,c_c,C_o,this]( const Int thread )
            {
                if constexpr( !valuesQ )
                {
                    (void)A_v;
                    (void)B_v;
                }
                if constexpr( !assemblerQ )
                {
                    (void)c_c;
                    (void)C_o;
                }
                const  Int i_begin = job_ptr[thread  ];
                const  Int i_end   = job_ptr[thread+1];
                
                const LInt new_pos = B_o[i_begin];
                const LInt     pos = A_o[i_begin];
                
                const LInt thread_nonzeroes = B_o[i_end] - B_o[i_begin];
                
                copy_buffer( &A_i[pos], &B_i[new_pos], thread_nonzeroes );
                if constexpr( valuesQ )
                {
                    copy_buffer( &A_v[pos], &B_v[new_pos], thread_nonzeroes );
                }
                if constexpr( assemblerQ )
                {
                    copy_buffer( &c_c[pos], &C_o[new_pos+LInt(1)], thread_nonzeroes );
                }
            },
            job_ptr.ThreadCount(), job_ptr.ThreadCount()
        );
    }

    swap(B_outer,A_outer);
    swap(B_inner,A_inner);
    if constexpr( valuesQ )
    {
        swap(B_values,A_values);
    }
    if constexpr( assemblerQ )
    {
        // TODO: Test what works better.
        C_outer.template Accumulate<parQ>(thread_count);
//        C_outer.Accumulate(); // I think I prefer the sequential code here.
    }

    job_ptr = JobPointers<Int>();
    job_ptr_initialized = false;
    proven_duplicate_freeQ = true;
}
