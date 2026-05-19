#pragma once

// No, I do not really consider to include this maintenance hell ever again.
//#include <omp.h>

// TODO: Improve load balancing.
// TODO: - Reorder `subrees` in `Tree` based on this cost estimate.


// TODO: Correct move and copy constructors...

// TODO: Would be great to allow nodes in the top levels of the tree to use more than one thread...

namespace Tensors
{
    template<SignedIntQ Int_, Parallel_T parQ_>
    class Tree final
    {
    public:
        
        using Int = Int_;
        
        static constexpr Parallel_T parQ = parQ_;
        
        using Permutation_T  = Permutation<Int,parQ>;
        using BinaryMatrix_T = Sparse::BinaryMatrixCSR<Int,Int,parQ>;
        
    protected:
        
        Int n; // The number of vertices, _including_ the virtual root vertex.
        
        Int thread_count;

        Tensor1<Int,Int> parents;
        Tensor1<Int,Int> node_to_level;
        Tensor1<Int,Int> node_to_split_level;
        
        Tensor1<double,Int> costs;
        Tensor1<double,Int> desc_costs;
        
        BinaryMatrix_T A;      // The adjacency matrix of the directed graph.
        
        Tensor1<Int,Int> desc_counts;
        
        Permutation<Int,parQ> post;                   // To store the postordering.
        
        BinaryMatrix_T levels; // The adjacency matrix of the directed graph.

        static constexpr Int zero = 0;
        static constexpr Int one  = 1;
        
        
        std::vector<std::vector<Int>> tree_top_levels;
        
        std::vector<Int> tree_top_vertices;
        
        std::vector<Int> subtrees;
        
        mutable Tensor1<Int,Int> check_list;

        Int max_depth = 0;
        Int max_split_depth = 0;
        
    public:
        
        explicit Tree(
            Tensor1<Int,Int> && parents_,
            const Int thread_count_ = 1
        )
        :   n                 ( parents_.Size()+1   )
        // We use an additional virtual vertex as root.
        ,   thread_count      ( thread_count_       )
        ,   parents           ( std::move(parents_) )
        ,   costs             ( n, 1.               )
        {
            Init();
        }
        
        Tree(
            Tensor1<   Int,Int> && parents_,
            Tensor1<double,Int> && costs_,
            const Int thread_count_ = 1
        )
        :   n                 ( parents_.Size()+1   )
        // We use an additional virtual vertex as root.
        ,   thread_count      ( thread_count_       )
        ,   parents           ( std::move(parents_) )
        ,   costs             ( std::move(costs_)   )
        {
            Init();
        }
        
    public:
        
        // Default constructor
        Tree() = default;
        // Destructor
        ~Tree() = default;
        // Copy constructor
        Tree( const Tree & other ) = default;
        // Copy assignment operator
        Tree & operator=( const Tree & other ) = default;
        // Move constructor
        Tree( Tree && other ) = default;
        // Move assignment operator
        Tree & operator=( Tree && other ) = default;
        
//        // Destructor
//        ~Tree() = default;
//        
//        // Copy constructor
//        Tree( const Tree & other )
//        :   n                 ( other.n                 )
//        ,   thread_count      ( other.thread_count      )
//        ,   parents           ( other.parents           )
//        ,   costs             ( other.costs             )
//        ,   desc_costs        ( other.desc_costs        )
//        ,   A                 ( other.A                 )
//        ,   desc_counts       ( other.desc_counts       )
//        ,   post              ( other.post              )
//        ,   levels            ( other.levels            )
//        ,   tree_top_levels   ( other.tree_top_levels   )
//        ,   tree_top_vertices ( other.tree_top_vertices )
//        ,   subtrees          ( other.subtrees          )
//        {}
//
//        // We could also simply use the implicitly created copy constructor.
//
//        friend void swap (Tree &X, Tree &Y ) noexcept
//        {
//            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//            using std::swap;
//
//            swap( X.n,                 Y.n                 );
//            swap( X.thread_count,      Y.thread_count      );
//            swap( X.parents,           Y.parents           );
//            swap( X.A,                 Y.A                 );
//            swap( X.desc_counts, Y.desc_counts );
//            swap( X.post,              Y.post              );
//            swap( X.costs,             Y.costs             );
//            swap( X.desc_costs,  Y.desc_costs  );
//            swap( X.levels,            Y.levels            );
//            swap( X.tree_top_vertices, Y.tree_top_vertices );
//            swap( X.subtrees,          Y.subtrees          );
//            swap( X.tree_top_levels,   Y.tree_top_levels   );
//        }
//
//        // Copy assignment operator
//        Tree & operator=(Tree other)
//        {
//            // copy-and-swap idiom
//            // see https://stackoverflow.com/a/3279550/8248900 for details
//
//            swap(*this, other);
//
//            return *this;
//        }
//
//        // Move constructor
//        Tree( Tree && other ) noexcept : Tree()
//        {
//            swap(*this, other);
//        }
        
        
        cref<Tensor1<Int,Int>> Parents() const
        {
            return parents;
        }
        
        cref<Permutation_T> PostOrdering() const
        {
            return post;
        }
        
        cref<Tensor1<Int,Int>> DescendantCounts() const
        {
            return desc_counts;
        }
        
        TOOLS_FORCE_INLINE Int DescendantCount( const Int i ) const
        {
            return desc_counts[i];
        }
        
        cref<Tensor1<Int,Int>> ChildPointers() const
        {
            return A.Outer();
        }
        
        TOOLS_FORCE_INLINE Int ChildPointer( const Int i ) const
        {
            return A.Outer(i);
        }
        
        cref<Tensor1<Int,Int>> ChildIndices() const
        {
            return A.Inner();
        }
        
        Int ChildIndex( const Int k ) const
        {
            return A.Inner(k);
        }
        
        cref<Tensor1<Int,Int>> LevelPointers() const
        {
            return levels.Outer();
        }
        
        TOOLS_FORCE_INLINE Int LevelPointer( const Int i ) const
        {
            return levels.Outer(i);
        }
        
        cref<Tensor1<Int,Int>> LevelIndices() const
        {
            return levels.Inner();
        }
        
        Int LevelIndex( const Int k ) const
        {
            return levels.Inner(k);
        }
        
        TOOLS_FORCE_INLINE Int VertexCount() const
        {
            return n;
        }
        
        TOOLS_FORCE_INLINE Int ChildCount( const Int i ) const
        {
            // Returns number of children of child i.
            return ChildPointer(i+1)-ChildPointer(i);
        }
        
        
        TOOLS_FORCE_INLINE Int Child( const Int i, const Int k ) const
        {
            // Returns k-th child of node i.
            return A.ChildIndex(ChildPointer(i)+k);
        }

        cref<Tensor1<double,Int>> Costs() const
        {
            return costs;
        }
        
        cref<Tensor1<double,Int>> DescendantCosts() const
        {
            return desc_costs;
        }
        
        cref<std::vector<Int>> TreeTopVertices() const
        {
            return tree_top_vertices;
        }
        
        cref<std::vector<Int>> SubtreeRoots() const
        {
            return subtrees;
        }
        
        bool PostOrderedQ() const
        {
            // TODO: Is this test good enough to indeed guarantee that the tree is postordered, when passed?
            return DoReduce<parQ>(
                  [=,this]( const Int i ) -> bool
                  {
                      const Int p_i = parents[i];
                      
                      //                    // This here would require signed integers.
                      //                    return
                      //                    (i < p_i)
                      //                    &&
                      //                    (i - DescendantCount(i) >= p_i - DescendantCount(p_i) );
                      
                      // This merely requires that 2 n fits into Int without overflow.
                      return
                      (i < p_i)
                      &&
                      (i + DescendantCount(p_i) >= p_i + DescendantCount(i) );
                  },
                  []( cref<bool> v, mref<bool> r ) { r = r && v; }, true,
                  Root(), thread_count
            );
            
            return true;
        }
        
        
        Int Root() const
        {
            return n-1;
        }
        
        
#include "Tree/Init.hpp"
#include "Tree/Traverse_PostOrdered_Sequential.hpp"
#include "Tree/Debugging.hpp"
        
#include "Tree/Traverse_PreOrdered.hpp"
#include "Tree/Traverse_PostOrdered.hpp"
//#include "Tree/Traverse_PostOrdered_OpenMP.hpp"
                 
    public:
        
        std::string ClassName() const
        {
            return std::string("Tree")+"<"+TypeName<Int>+">";
        }
    };
}
