protected:

template<bool assemblerQ>
void SortInner_impl(
    cptr<LInt> A_o,
    mptr< Int> A_i,
    mptr<Scal> A_v,
    mptr<LInt> A_f
) const
{
    if( proven_inner_sortedQ ) { return; }
    
    TOOLS_PTIMER(timer,ClassName()+"::SortInner_impl" + "<" + BoolString(assemblerQ) + ">");
    
    if( !this->WellFormedQ() ) { return; }

    RequireJobPtr();
    
    Do<parQ>(
        [=,this]( const Int thread )
        {
            std::conditional_t<assemblerQ,
                ThreeArraySort<Int,Scal,LInt,LInt>,
                TwoArraySort  <Int,Scal     ,LInt>
            > S;
            
            const Int i_begin = job_ptr[thread  ];
            const Int i_end   = job_ptr[thread+1];
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                const LInt begin = A_o[i  ];
                const LInt end   = A_o[i+1];
                
                if constexpr ( assemblerQ )
                {
                    S( &A_i[begin], &A_v[begin], &A_f[begin], end-begin );
                }
                else
                {
                    S( &A_i[begin], &A_v[begin], end-begin );
                }
            }
        },
        thread_count, thread_count
    );
    
    proven_inner_sortedQ = true;
}
