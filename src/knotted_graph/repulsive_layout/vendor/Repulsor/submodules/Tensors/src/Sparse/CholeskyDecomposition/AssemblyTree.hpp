public:
    
    const Tree_T & AssemblyTree()
    {
        SymbolicFactorization();
        
        return aTree;
    }
    
protected:

    void CreateAssemblyTree()
    {
        TOOLS_PTIMER(timer,ClassName()+"::CreateAssemblyTree");

        cref<Tensor1<Int,Int>> parents = EliminationTree().Parents();

        Tensor1<Int,Int> SN_parents ( SN_count );
        
        Tensor1<double,Int> SN_costs ( SN_count + 1 );

        constexpr double factor = 1./6.;
        
        for( Int k = 0; k < SN_count-1; ++k )
        {
            // This subtraction is safe as each supernode has at least one row.
            const Int last_row = SN_rp[k+1]-1;

            const Int last_rows_parent = parents[last_row];

            const Int parent = (last_rows_parent<n) ? row_to_SN[last_rows_parent] : SN_count;
            
            SN_parents[k] = parent;
            
            const Int n_0 =               SN_rp   [k+1] - SN_rp   [k];
            const Int n_1 = int_cast<Int>(SN_outer[k+1] - SN_outer[k]);
            
            // This is just the cost for `trsm` and `potrf` in `FactorizeSupernode`!
            
            SN_costs[k] = ( n_0 * ( n_0 + 1 ) ) * ( ( n_0 + 2 ) * factor + 0.5 * n_1 );
            
            // TODO: Add estimate for costs for `herk` and `gemm` in `FetchFromDescendants`!
            // TODO: Add estimate for costs for `ComputeIntersection`!
        }
        
        {
            const Int k = SN_count-1;
            
            SN_parents[SN_count-1] = SN_count;
            
            const Int n_0 =               SN_rp   [k+1] - SN_rp   [k];
            const Int n_1 = int_cast<Int>(SN_outer[k+1] - SN_outer[k]);
            
            SN_costs[k] = ( n_0 * ( n_0 + 1 ) ) * ( ( n_0 + 2 ) * factor + 0.5 * n_1 );
            
        }
        
        SN_costs[SN_count] = 0.;

        aTree = Tree_T ( std::move(SN_parents), std::move(SN_costs), thread_count );
    }
