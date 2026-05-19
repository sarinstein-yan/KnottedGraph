#pragma once

namespace Tensors
{
    namespace Dense
    {
        
        //        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        template<
            Size_T M_ct_,      Size_T N_ct_,      Size_T K_ct_,
            Size_T m_default_, Size_T n_default_, Size_T k_default_,
            Op opA, Op opB, Op opC,
            typename A_T_, typename B_T_, typename C_T_, IntQ Int_
        >
        class BLAS_3 final
        {
        public:
            
            using A_T = A_T_;
            using B_T = B_T_;
            using C_T = C_T_;
            using Int = Int_;
            
            static constexpr Int M_ct = M_ct_;
            static constexpr Int N_ct = N_ct_;
            static constexpr Int K_ct = K_ct_;
            
            static constexpr Int m_default = m_default_;
            static constexpr Int n_default = n_default_;
            static constexpr Int k_default = k_default_;
            
        private:
            
            //        static constexpr Int M_treshold = 8;
            //        static constexpr Int N_treshold = 8;
            //        static constexpr Int K_treshold = 8;
            
            static constexpr Int M_treshold = 1;
            static constexpr Int N_treshold = 1;
            static constexpr Int K_treshold = 1;
            
            // If any of the dimensions is prescibed
            
            static constexpr bool M_fixed_Q = (VarSize < M_ct);
            static constexpr bool N_fixed_Q = (VarSize < N_ct);
            static constexpr bool K_fixed_Q = (VarSize < K_ct);
            
            static constexpr bool M_small_Q = M_fixed_Q && (M_ct < M_treshold);
            static constexpr bool N_small_Q = N_fixed_Q && (N_ct < N_treshold);
            static constexpr bool K_small_Q = K_fixed_Q && (K_ct < K_treshold);
            
            static constexpr Int m = M_small_Q ? M_ct : m_default;
            static constexpr Int n = N_small_Q ? N_ct : n_default;
            static constexpr Int k = K_small_Q ? K_ct : k_default;
            
//            static constexpr Int mk = m * k;
//            static constexpr Int kn = k * n;
//            static constexpr Int mn = m * n;
            
            static constexpr Int a_size = m*k;
            static constexpr Int b_size = k*n;
            static constexpr Int c_size = m*n;
            
            const Int M_block_count_ct = CeilDivide( M_ct, m );
            const Int N_block_count_ct = CeilDivide( N_ct, n );
            const Int K_block_count_ct = CeilDivide( K_ct, k );
            
            const Int M_full_block_count_ct = FloorDivide( M_ct, m );
            const Int N_full_block_count_ct = FloorDivide( N_ct, n );
            const Int K_full_block_count_ct = FloorDivide( K_ct, k );
            
            
            using A_Storage_T = MatrixBlockMajor<M_ct,K_ct,m,k,Op::Id   ,opA,A_T,Int>;
            using B_Storage_T = MatrixBlockMajor<N_ct,K_ct,k,n,Op::Trans,opB,B_T,Int>;
            using C_Storage_T = MatrixBlockMajor<M_ct,N_ct,m,n,Op::Id   ,opC,C_T,Int>;
            
            using A_Block_T = A_Storage_T::Block_T;
            using B_Block_T = B_Storage_T::Block_T;
            using C_Block_T = C_Storage_T::Block_T;
                        
            using a_T = mat_T<k,m,A_T>;
            using b_T = mat_T<n,k,B_T>;
            using c_T = mat_T<n,m,C_T>;
            
        private:
            
            // Sizes at runtime.
            
            const Int M_rt;
            const Int N_rt;
            const Int K_rt;
            
            const Int thread_count = 1;
            
        public:
            
            // Buffers for the block-major format.
            
            A_Storage_T AP;
            B_Storage_T BP;
            C_Storage_T CP;
            
        public:
            
            BLAS_3( const Int M_ = M_ct, const Int N_ = N_ct, const Int K_ = K_ct, const Int thread_count_ = 1 )
            :   M_rt                    ( M_ )
            ,   N_rt                    ( N_ )
            ,   K_rt                    ( K_ )
            ,   thread_count            ( thread_count_ )
            ,   AP                      ( M_, K_, thread_count )
            ,   BP                      ( K_, N_, thread_count )
            ,   CP                      ( M_, N_, thread_count )
            {}
            
            
            // No default constructor
            BLAS_3() = default;
            // Destructor
            ~BLAS_3() = default;
            // Copy constructor
            BLAS_3( const BLAS_3 & other ) = default;
            // Copy assignment operator
            BLAS_3 & operator=( const BLAS_3 & other ) = default;
            // Move constructor
            BLAS_3( BLAS_3 && other ) = default;
            // Move assignment operator
            BLAS_3 & operator=( BLAS_3 && other ) = default;
            
        public:
            
            Int M_FullBlockCount() const
            {
                return AP.M_FullBlockCount();
            }
            
            Int K_BlockCount() const
            {
                return AP.N_FullBlockCount();
            }
            
            Int N_BlockCount() const
            {
                return CP.N_FullBlockCount();
            }

            
            
            Int M_BlockCount() const
            {
                return AP.M_BlockCount();
            }
            
            Int K_FullBlockCount() const
            {
                return AP.N_BlockCount();
            }
            
            Int N_FullBlockCount() const
            {
                return CP.N_BlockCount();
            }
            
            
            
        public:
            
#include "BLAS_3/DotBlocks.hpp"
//#include "BLAS_3/DotBlocks_mat_T.hpp"
//#include "BLAS_3/DotBlocks_with_pointers.hpp"
            
//#include "BLAS_3/DotBlocksRecursive.hpp"
#include "BLAS_3/gemm.hpp"

            
        public:
            
            std::string ClassName() const
            {
                return std::string("Dense::BLAS_3") +"<"
                + ToString(M_ct) +  "," + ToString(N_ct) + "," + ToString(K_ct) + ","
                + ToString(m) +  "," + ToString(n) + "," + ToString(k) + ","
                + TypeName<A_T> +  "," + TypeName<B_T> +  "," + TypeName<C_T> +  ","
                + ">(" + ToString(M_rt) + "," + ToString(N_rt) +  ")";
            }
            
        }; // class BLAS_3
        
    } // namespace Dense
}
