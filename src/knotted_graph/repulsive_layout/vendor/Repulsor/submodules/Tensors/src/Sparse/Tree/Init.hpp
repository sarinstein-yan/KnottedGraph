protected:
    
    // TODO: We need not always to create subtrees and tree tops...

    void Init()
    {
        TOOLS_PTIMER(timer,ClassName()+"::Init");
        
        Int target_split_level = thread_count <= 1 ? 1 : static_cast<Int>( std::ceil(std::log2( static_cast<double>(thread_count) )) + 3 );
        
        // First we build the adjacency matrix A of the tree.
        post = Permutation_T( n-1, thread_count );
        
        {
            Int list_count       = 1;
            Int entry_counts [1] = {n-1};
            
            const Int * idx_data = parents.data();
            const Int * jdx_data = post.GetPermutation().data(); // This is a iota.
            
            A = BinaryMatrix_T(
                &idx_data,
                &jdx_data,
                &entry_counts[0],
                list_count, n, n, thread_count, false, 0
            );
        }
        
        
        // Now we compute postordering, desc counts and some auxiliary data for later traversals.
        Int counter = 0;
        
        
        subtrees          = std::vector<Int>();
        tree_top_vertices = std::vector<Int>();
        tree_top_levels   = std::vector<std::vector<Int>>(1);
        
//        tree_top_vertices.push_back(Root());
//        tree_top_levels[0].push_back(Root());
        
        
        node_to_level       = Tensor1<Int,Int>( n, 0 );
        node_to_split_level = Tensor1<Int,Int>( n, 0 );
        
        node_to_level      [Root()] = 0;
        node_to_split_level[Root()] = 0;
        
        max_depth = 0;
        max_split_depth = 0;

        desc_costs  = Tensor1<double,Int> ( n );
        desc_counts = Tensor1<Int,Int>    ( n );
        
        
        mptr<Int> p = post.Scratch().data();
        
        const Int root = Root();
        
        Traverse_PostOrdered_Sequential(
            [this]( const Int node )
            {
                const Int k_begin = ChildPointer( node     );
                const Int k_end   = ChildPointer( node + 1 );
                
                const Int next_level = node_to_level[node] + 1;
                

                max_depth = Max(max_depth, next_level);

                Int next_split_level = node_to_split_level[node];
                
                if( k_end > k_begin + 1 )
                {
                    ++next_split_level;
                }
                
                for( Int k = k_begin; k < k_end; ++k )
                {
                    const Int child = ChildIndex(k);
                    
                    node_to_level       [child] = next_level;
                    node_to_split_level [child] = next_split_level;
                }
                
                return true;
            }
            ,
            [=,this,&counter]( const Int node )
            {
                if( counter < root )
                {
                    p[counter] = node;
                }
                
                ++counter;
                
                const Int split_level = node_to_split_level [node];
                
                const Int k_begin = ChildPointer( node     );
                const Int k_end   = ChildPointer( node + 1 );
                
                // The node itself is _not_ counted as its own desc!
                Int sum = 0;
                
                double cost = 0.; // costs[node] has already been filled.
                
                for( Int k = k_begin; k < k_end; ++k )
                {
                    const Int child = ChildIndex(k);
                    sum  += desc_counts[child] + 1;
                    cost += desc_costs [child] + costs[child];
                    
                    const Int child_level       = node_to_level[child];
                    const Int child_split_level = node_to_split_level[child];
                    
                    if( child_split_level == target_split_level )
                    {
                        if( child_split_level > split_level )
                        {
                            subtrees.push_back(child);
                        }
                    }
                    else if (child_split_level < target_split_level )
                    {
                        tree_top_vertices.push_back(child);
                        
                        
                        while( child_level >= static_cast<Int>(tree_top_levels.size()) )
                        {
                            tree_top_levels.emplace_back();
                        }
                        tree_top_levels[static_cast<Size_T>(child_level)].push_back(child);
                        
                    }
                        
                }
                desc_counts[node]  = sum;
                desc_costs [node]  = cost;
            }
            ,
            []( const Int node )
            {
                (void)node;
            }
        );
        
        // Build the levels matrix (is this needed at all?)
        {
            Int list_count       = 1;
            Int entry_counts [1] = {n-1};

            const Int * idx_data = node_to_level.data();
            const Int * jdx_data = post.GetPermutation().data(); // This is still a iota.

            levels = BinaryMatrix_T(
                &idx_data,
                &jdx_data,
                &entry_counts[0],
                list_count, max_depth+1, n, thread_count, true, 0
            );
        }

        // Now post.Scratch() contains the post ordering.
        post.SwapScratch( Inverse::False );
        
        // Now post.GetPermutation() contains the post ordering.
    }
