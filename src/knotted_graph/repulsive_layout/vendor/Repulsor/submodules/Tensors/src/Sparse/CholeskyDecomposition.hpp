#pragma once

#include "../BLAS_Wrappers.hpp"
#include "../LAPACK_Wrappers.hpp"

#include "CholeskyDecomposition/LeftLooking.hpp"
#include "CholeskyDecomposition/Multifrontal.hpp"
#include "CholeskyDecomposition/UpperSolver.hpp"
#include "CholeskyDecomposition/LowerSolver.hpp"

// Priority I:

// TODO: Compute AMD reordering. Often it works very well!

// TODO: LowerSolve seems to be unneccessary slow for nrhs = 1 and thread_count = 1.

// TODO: Improve scheduling for parallel factorization.
// TODO: - What to do if top of the tree is not a binary tree?
// DONE: - What to do in case of a forest?
// TODO: - Estimate work to do in subtrees.
// TODO: - Reorder `subrees` in `Tree` based on this cost estimate.

// Priority II:

// TODO: Parallelize symbolic factorization.
// TODO: - Build aTree first and traverse it in parallel to determine SN_inner.

// TODO: Do we really have to build _two_ EliminationTrees?
// TODO: Can we parallelize EliminationTrees? E.g., build it for chunks of rows and then merge the trees somehow?

// TODO: Improve solve phase:
// TODO: - trsm/trsv like interface
// TODO: - Fixed size arithmetic?
// TODO: - leading dimensions
// TODO: - multiplication and add-into

// TODO: Compute nested dissection --> Metis, Scotch. Parallel versions? MT-Metis?


// Priority III:

// TODO: hierarchical low-rank factorization of supernodes?
//      --> Superfast Multifrontal Method for Large Structured Linear Systems of Equations
//
// TODO: Iterative refinement -> CG solver?

// Priority IV:
// TODO: incomplete factorization?

// TODO: Maybe load linear combination of matrices A (with sub-pattern, of course) during factorization?

// TODO: parallelize potrf + trsm + herk of large supernodes.
// DONT: not helpful if Apple Accelerate is used?!?


// TODO: Speed up supernode update in left-looking factorization phase.
// TODO: - transpose U_0 and U_1 to reduce scatter_reads/scatter_adds.
// DONT: - fetching updates from descendants can be done in parallel


// DON'Ts:

// DONT: Allow the user to supply only upper or lower triangle of matrix.
//           --> I think this feature is seldomly used and creates some bad incentives.
//           --> iterative refinement would need the whole matrix anyways to be fast.

// DONT: employ Tiny::BLAS kernels. --> Does not seem to be helpful...
//           --> Did not help; at least on M1, the BLAS kernels are fast also for small sizes.


// Super helpful literature:
// Stewart - Building an Old-Fashioned Sparse Solver. http://hdl.handle.net/1903/1312

// TODO: On Computing Min-Degree Elimination Orderings. https://arxiv.org/pdf/1711.08446.pdf:

// TODO: Kayaaslan, Ucar - Reducing elimination tree height for parallel LU factorization of sparse unsymmetric matrices. https://hal.inria.fr/hal-01114413/document

// TODO: Liu - The Multifrontal Method for Sparse Matrix Solution: Theory and Practice. https://www.jstor.org/stable/2132786 !!

namespace Tensors
{
    namespace Sparse
    {
        template<typename Scal_, IntQ Int_, IntQ LInt_, Parallel_T parQ_>
        class alignas( ObjectAlignment ) CholeskyDecomposition final : public CachedObject<1,1,1,1>
        {
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
            static constexpr Parallel_T parQ = parQ_;
            
            using BinaryMatrix_T     = Sparse::BinaryMatrixCSR<Int,LInt,parQ>;
            using Matrix_T           = Sparse::MatrixCSR<Scal,Int,LInt,parQ>;
            using Tree_T             = Tree<Int,parQ>;
            using Permutation_T      = Permutation<Int,parQ>;
            
            using Factorizer_LL_T    = CholeskyFactorizer_LeftLooking<Scal,Int,LInt>;
            using Factorizer_MF_T    = CholeskyFactorizer_Multifrontal<Scal,Int,LInt>;
            
            using Update_T           = Tensor2<Scal,LInt>;
//            using Update_T           = Scal *;
            
            friend Factorizer_LL_T;
            friend Factorizer_MF_T;
            
            friend class UpperSolver<false,Scal,Int,LInt>;
            friend class UpperSolver<true, Scal,Int,LInt>;
            friend class LowerSolver<false,false,Scal,Int,LInt>;
            friend class LowerSolver<true, false,Scal,Int,LInt>;
            friend class LowerSolver<false,true, Scal,Int,LInt>;
            friend class LowerSolver<true, true, Scal,Int,LInt>;
            
            using VectorContainer_T = Tensor1<Scal,LInt>;

            enum struct SupernodeStrategy_T : Int8
            {
                Maximal     = 0,
                Fundamental = 1,
                Amalgamated = 2
            };
            
            enum struct FactorizationMethod_T : Int8
            {
                Multifrontal = 0,
                LeftLooking  = 1
            };
            
            class NumericalFactorization_T
            {
            public:
                
                friend class CholeskyDecomposition<Scal,Int,LInt,Parallel>;
                friend class CholeskyDecomposition<Scal,Int,LInt,Sequential>;
                friend class UpperSolver<false,Scal,Int,LInt>;
                friend class UpperSolver<true, Scal,Int,LInt>;
                friend class LowerSolver<false,false,Scal,Int,LInt>;
                friend class LowerSolver<true, false,Scal,Int,LInt>;
                friend class LowerSolver<false,true, Scal,Int,LInt>;
                friend class LowerSolver<true, true, Scal,Int,LInt>;
                friend class CholeskyFactorizer_LeftLooking<Scal,Int,LInt>;
                friend class CholeskyFactorizer_Multifrontal<Scal,Int,LInt>;
                
            private:
                
                Tensor1<Scal,LInt> tri_val;
                Tensor1<Scal,LInt> rec_val;

                bool factorizedQ = false;
                bool goodQ       = false;
                
            public:
                
                NumericalFactorization_T() = default;
                
                NumericalFactorization_T(
                    const LInt tri_size, const LInt rec_size
                )
                :   tri_val ( tri_size )
                ,   rec_val ( rec_size )
                {}
                
                bool NumericallyFactorizedQ() const
                {
                    return factorizedQ;
                }
                
                bool NumericallyGoodQ() const
                {
                    return goodQ;
                }
                
                LInt TriangularValueSize() const
                {
                    return tri_val.Size();
                }
                
                LInt RectangularValueSize() const
                {
                    return rec_val.Size();
                }
                
                friend void swap(
                    NumericalFactorization_T && A_,
                    NumericalFactorization_T && B_
                ) noexcept
                {
                    swap( A_.tri_val,       B_.tri_val      );
                    swap( A_.rec_val,       B_.rec_val      );
                    swap( A_.factorizedQ,   B_.factorizedQ  );
                    swap( A_.goodQ,         B_.goodQ        );
                }
                
                Size_T AllocatedByteCount() const
                {
                    return sizeof(Scal) * ToSize_T(tri_val.Size() + rec_val.Size());
                }
            };
            
        protected:
            
            static constexpr Int izero = 0;
            static constexpr Int ione  = 1;
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;
            
            Int n = 0;
            Int thread_count = 1;
            
            Permutation_T perm; // row and column permutation of the nonzeros of the matrix.
            
            BinaryMatrix_T A;
            
            Tensor1<LInt,LInt> A_inner_perm; // permutation of the nonzero values. Needed for reading in.
            
            Tensor1<Scal,LInt> A_val;
            Scal reg = 0;
            
            // elimination tree
            bool eTree_initializedQ = false;
            Tree_T eTree;
            
            // assembly tree
            Tree_T aTree;
            
            // Supernode data:
            
            Int  amalgamation_threshold = 4;
            bool SN_initializedQ        = false;
            
            SupernodeStrategy_T SN_strategy = SupernodeStrategy_T::Maximal;
            // Number of supernodes.
            Int SN_count = 0;
            
            // Pointers from supernodes to their rows.
            // k-th supernode has rows [ SN_rp[k],SN_rp[k]+1,...,SN_rp[k+1] [
            Tensor1< Int, Int> SN_rp;
            // Pointers from supernodes to their starting position in SN_inner.
            Tensor1<LInt, Int> SN_outer;
            // The column indices of rectangular part of the supernodes.
            Tensor1< Int,LInt> SN_inner;
            
            // Hence, the k-th supernode has the following column indices:
            // triangular  part = [ i_begin, i_begin+1,...,i_end [
            // rectangular part = [
            //                      SN_inner[j_begin  ],
            //                      SN_inner[j_begin+1],
            //                      SN_inner[j_begin+2],
            //                      ...,
            //                      SN_inner[j_end]
            //                    [
            // where i_begin = SN_rp[k],
            //       i_end   = SN_rp[k+1],
            //       j_begin = SN_outer[k], and
            //       j_end   = SN_outer[SN_outer[k+1]].
            
            // i-th row of U belongs to supernode row_to_SN[i].
            Tensor1< Int, Int> row_to_SN;
            
            // Hence, the column indices of U for row i can are:
            // triangular  part = [ i_begin,i_begin+1,...,i_end [
            // rectangular part = [
            //                      SN_inner[j_begin  ],
            //                      SN_inner[j_begin+1],
            //                      SN_inner[j_begin+2],
            //                      ...,
            //                      SN_inner[j_end]
            //                    [
            // where i_begin = SN_rp[row_to_SN[i]] = i,
            //       i_end   = SN_rp[row_to_SN[i]+1],
            //       j_begin = SN_outer[row_to_SN[i]  ], and
            //       j_end   = SN_outer[row_to_SN[i]+1].
            
            // Values of triangular part of s-th supernode is stored in
            // [ SN_data.tri_val[SN_tri_ptr[s]],...,SN_data.tri_val[SN_tri_ptr[s]+1] [
            Tensor1<LInt, Int> SN_tri_ptr;
            
            // Values of rectangular part of s-th supernode is stored in
            // [ SN_data.rec_val[SN_rec_ptr[s]],...,SN_data.rec_val[SN_rec_ptr[s]+1] [,
            Tensor1<LInt, Int> SN_rec_ptr;
            
            NumericalFactorization_T SN_data;
            
            FactorizationMethod_T factorization_method = FactorizationMethod_T::Multifrontal;

            
            std::vector<Update_T> SN_updates;
            
            // Maximal size of triangular part of supernodes.
            Int max_n_0 = 0;
            // Maximal size of rectangular part of supernodes.
            Int max_n_1 = 0;
            
            Int nrhs = 1;
            
            // Stores right hand side / solution during the solve phase.
            VectorContainer_T X;
            
            // Stores right hand side / solution during the solve phase.
            // Some scratch space to read parts of X that belong to a supernode's rectangular part.
            VectorContainer_T X_scratch;
            // TODO: If I want to parallelize the solve phase, I have to provide each thread with its own X_scratch.
            
            std::vector<std::mutex> row_mutexes;
            
        public:
            
            template<IntQ ExtLInt, IntQ ExtInt>
            CholeskyDecomposition(
                cptr<ExtLInt> A_outer,
                cptr<ExtInt>  A_inner,
                Permutation_T && perm_
            )
            :   n               ( Max( izero, perm_.Size() )      )
            ,   thread_count    ( Max( ione, perm_.ThreadCount()) )
            {
                TOOLS_PTIMER(timer,ClassName()+"( "+ TypeName<ExtLInt> + "*, "+ TypeName<ExtInt> + "*,  Permutation<" + TypeName<Int>+ "> )");
                
                perm = Permutation_T( std::move( perm_) );
                
                A = BinaryMatrix_T( A_outer, A_inner, n, n, thread_count );
                
                A_val = Tensor1<Scal,LInt>( A.NonzeroCount() );
                
                A_inner_perm = A.Permute( perm, perm );
                Init();
            }
            
            // This is the constructor that will most likely to be used in practice.
            // p is supposed to be a vector of size n_ containing a fill-in reducing permutation of [0,...,n[.
            template<IntQ ExtLInt, IntQ ExtInt, IntQ ExtInt2>
            CholeskyDecomposition(
                cptr<ExtLInt> A_outer,
                cptr<ExtInt>  A_inner,
                cptr<ExtInt2> p,
                Int n_, Int thread_count_
            )
            :   CholeskyDecomposition(
                    A_outer, A_inner, Permutation_T( p, n_, Inverse::False, thread_count_ )
                )
            {}
            
            // Constructor if the user has applied a fill-in
            // reducing permutation to the matrix already.
            template<IntQ ExtLInt, IntQ ExtInt>
            CholeskyDecomposition(
                cptr<ExtLInt> A_outer,
                cptr<ExtInt>  A_inner,
                Int n_, Int thread_count_
            )
            :   n               ( n_                        )
            ,   thread_count    ( Max( ione, thread_count_) )
            {
                TOOLS_PTIMER(timer,ClassName()+"( "+ TypeName<ExtLInt> + "*, "+ TypeName<ExtInt> + "*, " + TypeName<Int>+ ", " + TypeName<Int>+ " )");
                
                A = BinaryMatrix_T( A_outer, A_inner, n, n, thread_count );
                
                A_val = Tensor1<Scal,LInt>( A.NonzeroCount() );

                perm = Permutation_T( n_, thread_count ); // use identity permutation
                    
                A_inner_perm = Tensor1<LInt,LInt>( A.NonzeroCount() );

                A_inner_perm.iota( thread_count );

                Init();
            }

            // Default constructor
            CholeskyDecomposition() = default;
            // Destructor (virtual because of inheritance)
            virtual ~CholeskyDecomposition() override = default;
            // Copy constructor
            CholeskyDecomposition( const CholeskyDecomposition & other ) = default;
            // Copy assignment operator
            CholeskyDecomposition & operator=( const CholeskyDecomposition & other ) = default;
            // Move constructor
            CholeskyDecomposition( CholeskyDecomposition && other ) = default;
            // Move assignment operator
            CholeskyDecomposition & operator=( CholeskyDecomposition && other ) = default;

            
        protected:
            
            void Init()
            {
                TOOLS_PTIMER(timer,ClassName()+"::Init");
                if( n <= izero )
                {
                    eprint(ClassName()+": Size n = "+ToString(n)+" of matrix is <= 0.");
                }
                A.RequireDiag();
                CheckDiagonal();
                row_mutexes = std::vector<std::mutex> ( static_cast<Size_T>(n) );
            }
            
            
        public:
            
            void CheckDiagonal() const
            {
                TOOLS_PTIMER(timer,ClassName()+"CheckDiagonal()");
                
                A.RequireDiag();
                
                bool okay = true;
                
                for( Int i = 0; i < n; ++i )
                {
                    okay = okay && (A.Inner(A.Diag(i)) == i);
                }
                
                if( !okay )
                {
                    eprint(ClassName()+"::PostOrdering: Diagonal of input matrix is not marked as nonzero.");
                    TOOLS_DUMP( A.Diag() );
                    
                }
            }
            
        protected:
            
            cref<Tensor1<Int,Int>> PostOrdering()
            {
                cref<Permutation_T> post = EliminationTree().PostOrdering();
                
                if( !EliminationTree().PostOrderedQ() )
                {
                    TOOLS_PTIMER(timer,ClassName()+"::PostOrdering");
                    
                    perm.Compose( post, Compose::Post );
                    
                    
                    // `post` will reorder the inner indices; hence, we have to reorder also `A_inner_perm`;
                    // Otherwise, `Factorizer_LL_T` will read the wrong nonzero values.
                    {
                        Tensor1<LInt,LInt> inner_perm_perm = A.Permute( post, post );
                    
                        // A_inner_perm.Compose( std::move(A.Permute( post, post )), Compose::Post );
                        
                        cptr<LInt> p = A_inner_perm.data();
                        mptr<LInt> q = inner_perm_perm.data();
                        
                        Do<parQ>(
                            [p,q]( const LInt i ) { q[i] = p[q[i]]; },
                            A_inner_perm.Size(), static_cast<LInt>(thread_count)
                        );
                        
                        swap(A_inner_perm,inner_perm_perm);
                    }
                    
                    
                    A.RequireDiag();
                    
                    CheckDiagonal();
                    
                    eTree_initializedQ  = false;
                    SN_initializedQ     = false;
                    SN_data.factorizedQ = false;
                    SN_data.goodQ       = false;
                    
                    // TODO:  Is there a cheaper way to generate the correct tree,
                    // TODO:  e.g., by permuting the old tree?
                    (void)EliminationTree();
                }
                
                return EliminationTree().PostOrdering().GetPermutation();
            }
            
#include "CholeskyDecomposition/EliminationTree.hpp"
#include "CholeskyDecomposition/AssemblyTree.hpp"
#include "CholeskyDecomposition/InputOutput.hpp"
#include "CholeskyDecomposition/Symbolic.hpp"
#include "CholeskyDecomposition/Numeric.hpp"
#include "CholeskyDecomposition/Solve.hpp"
    
        public:
            
            static std::string MethodName( const std::string & tag )
            {
                return ClassName() + "::" + tag;
            }
            
            static std::string ClassName()
            {
                return std::string("Sparse::CholeskyDecomposition")+"<"+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }

            
        }; // class CholeskyDecomposition
        
    } // namespace Sparse
    
//    template<typename Scal, IntQ Int, IntQ LInt>
//    std::string ToString( cref<typename Sparse:: CholeskyDecomposition<Scal,Int,LInt>::SupernodeStrategy_T> s
//    )
//    {
//        using S_T = typename  Sparse::CholeskyDecomposition<Scal,Int,LInt>::SupernodeStrategy_T;
//        
//        switch ( s )
//        {
//            case S_T::Maximal:
//            {
//                return "Maximal";
//            }
//            case S_T::Fundamental:
//            {
//                return "Fundamental";
//            }
//            case S_T::Amalgamated:
//            {
//                return "Amalgamated";
//            }
//            default:
//            {
//                return "Unknown";
//            }
//        }
//    }
//    
//    template<typename Scal, IntQ Int, IntQ LInt>
//    std::string ToString( cref<typename Sparse::CholeskyDecomposition<Scal,Int,LInt>::FactorizationMethod_T> m
//    )
//    {
//        using M_T = typename  Sparse::CholeskyDecomposition<Scal,Int,LInt>::FactorizationMethod_T;
//        
//        switch ( m )
//        {
//            case M_T::Multifrontal:
//            {
//                return "Multifrontal";
//            }
//            case M_T::LeftLooking:
//            {
//                return "LeftLooking";
//            }
//            default:
//            {
//                return "Unknown";
//            }
//        }
//    }
    
} // namespace Tensors





// =========================================================
// DONE: Reordering in the solve phase.
//          --> Copy-cast during pre- and post-permutation.
//          --> ReadRightHandSide, WriteSolution

// DONE: Load A + eps * Id during factorization.

// DONE:Call SymbolicFactorization, NumericFactorization,... when dependent routines are called.

// DONE: Currently, EliminationTree breaks down if the matrix is reducible.
//           --> What we need is an EliminationForest!
//           --> Maybe it just suffices to append a virtual root (that is not to be factorized).



// DONE: Automatically determine postordering and apply it!

// DONE: Allow the user to supply a permutation.

// DONE: Parallelized, abstract postorder traversal of Tree

// DONE: Specialization of the cases m_0 = 1 and n_0 = 1.

// DONE: Return permutation (as sparse matrices) so that they can be checked. (< 2023-06-25)

// DONE: Return factors (as sparse matrices) so that they can be checked. (< 2023-06-25)

// DONE: Parallelize upper solve phase. (2023-06-26)

// DONE: Parallelize lower solve phase. (2032-07-30)

// DONE: User interface for lower/upper solves. (2032-07-30)

// DONE: A_inner_perm seems to be a bit wasteful; we neither need the inverse permutation nor scratch. (2032-07-30)

// DONE: Skip some unrelevant descendants in left-looking factorization.
