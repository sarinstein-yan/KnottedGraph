#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        template<Size_T M, Size_T N, Size_T K, AddTo_T addto, 
            typename A_T, typename B_T, typename C_T
        >
        void fixed_dot_mm_naive( cptr<A_T> A_, cptr<B_T> B_, mptr<C_T> C_ )
        {
            // A is of size M X K
            // B is of size K X N
            // C is of size M X N
            
            // All matrices are assumed to be in row-major storage.
            
            auto A = [=]( Size_T i, Size_T k ) -> A_T   { return A_[K*i+k]; };
            
            auto B = [=]( Size_T k, Size_T j ) -> B_T   { return B_[N*k+j]; };
            
            auto C = [=]( Size_T i, Size_T j ) -> C_T & { return C_[N*i+j]; };
            
            if constexpr ( addto == Tensors::AddTo )
            {
                for( Size_T k = 0; k < K; ++k )
                {
                    for( Size_T i = 0; i < M; ++i )
                    {
                        for( Size_T j = 0; j < N; ++j )
                        {
                            C(i,j) += A(i,k) * B(k,j);
                        }
                    }
                }
            }
            else
            {
                // First pass to overwrite (if desired).
                {
                    constexpr Size_T k = 0;
                    
                    for( Size_T i = 0; i < M; ++i )
                    {
                        for( Size_T j = 0; j < N; ++j )
                        {
                            C(i,j) = A(i,k) * B(k,j);
                        }
                    }
                    
                }
                // Now add-in the rest.
                for( Size_T k = 1; k < K; ++k )
                {
                    for( Size_T i = 0; i < M; ++i )
                    {
                        for( Size_T j = 0; j < N; ++j )
                        {
                            C(i,j) += A(i,k) * B(k,j);
                        }
                    }
                }
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors

