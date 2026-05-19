public:

// Sequential postorder traversal. Meant to be used only for initialization and for reference purposes.
template<class PreVisit_T, class PostVisit_T, class LeafVisit_T>
void Traverse_PostOrdered_Sequential(
    PreVisit_T  && pre_visit,
    PostVisit_T && post_visit,
    LeafVisit_T && leaf_visit,
    const Int tree_top_depth = std::numeric_limits<Int>::max()
)
{
    TOOLS_PTIMER(timer,ClassName()+"::Traverse_PostOrdered_Sequential");
    
    Tensor1<Int, Int> stack   ( n );
    Tensor1<Int, Int> depth   ( n );
    Tensor1<bool,Int> visited ( n, false );

    Int i = zero; // stack pointer
    
    stack[i] = Root();
    depth[i] = zero;
    
    // post order traversal of the tree
    while( i >= zero )
    {
        const Int node    = stack[i];
        const Int d       = depth[i];
        const Int k_begin = ChildPointer(node  );
        const Int k_end   = ChildPointer(node+1);
        
        if( !visited[i] && (d < tree_top_depth) && (k_begin < k_end) )
        {
            // TODO: Add a check for whether pre_visit returns true.
            // We visit this node for the first time.
            (void)pre_visit(node);
            
            visited[i] = true;
            
            // Pushing the children in reverse order onto the stack.
            for( Int k = k_end; k --> k_begin; )
            {
                stack[++i] = ChildIndex(k);
                depth[i]   = d+1;
            }
            
//                    // Pushing the children in forward order onto the stack.
//                    for( Int k = k_begin; k < k_end; ++k )
//                    {
//                        stack[++i] = ChildIndex(k);
//                        depth[i]   = d+1;
//                    }
        }
        else
        {
            // Visiting the node for the second time.
            // We are moving in direction towards the root.
            // Hence, all children have already been visited.
            
            // Popping current node from the stack.
            visited[i--] = false;

            // things to be done when node is a leaf.
            if (k_begin == k_end)
            {
                leaf_visit(node);
            }
            
            post_visit(node);
        }
    }
}
