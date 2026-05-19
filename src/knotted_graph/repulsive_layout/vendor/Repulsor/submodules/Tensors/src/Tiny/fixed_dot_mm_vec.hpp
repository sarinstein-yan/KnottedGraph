#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        template<Size_T M, Size_T N, Size_T K, AddTo_T addto,
            typename A_T, typename B_T, typename C_T
        >
        void fixed_dot_mm_vec( cptr<A_T> A_, cptr<B_T> B_, mptr<C_T> C_ )
        {
            // A is of size M X K
            // B is of size K X N
            // C is of size M X N
            
            // All matrices are assumed to be in row-major storage.
            
            if constexpr ( VectorizableQ<C_T> )
            {
                auto A = [=]( Size_T i, Size_T k ) -> C_T 
                {
                    return static_cast<C_T>(A_[K*i+k]);
                };
                
                using Row_T = vec_T<N,C_T>;
                
                Row_T B [K];
                Row_T C [M];
                
                copy_buffer<K*N>( B_, get_ptr(&B[0]) );
                
                // First pass to overwrite (if desired).
                {
                    constexpr Size_T k = 0;
                    
                    for( Size_T i = 0; i < M; ++i )
                    {
                        if constexpr ( addto == Tensors::AddTo )
                        {
                            C[i] += A(i,k) * B[k];
                        }
                        else
                        {
                            C[i] = A(i,k) * B[k];
                        }
                    }
                }
                
                // Now add-in the rest.
                for( Size_T k = 1; k < K; ++k )
                {
                    for( Size_T i = 0; i < M; ++i )
                    {
                        C[i] += A(i,k) * B[k];
                    }
                }

                copy_buffer<M*N>( get_ptr(&C[0]), C_ );
            }
            else // Generic fallback method.
            {
                
                auto A = [=]( Size_T i, Size_T k ) -> A_T
                {
                    return static_cast<C_T>(A_[K*i+k]);
                };
                
                auto B = [=]( Size_T k ) -> const B_T *
                {
                    return &B_[N*k];
                };
                
                auto C = [=]( Size_T i ) -> C_T *
                {
                    return &C_[N*i];
                };
                
            
                // First pass to overwrite (if desired).
                {
                    constexpr Size_T k = 0;
                    
                    for( Size_T i = 0; i < M; ++i )
                    {
                        if constexpr ( addto == Tensors::AddTo )
                        {
                            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,N>(
                                A(i,k), B(k), Scalar::One<C_T>, C(i)
                            );
                        }
                        else
                        {
                            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,N>(
                                A(i,k), B(k), Scalar::Zero<C_T>, C(i)
                            );
                        }
                    }
                }
    
                // Now add-in the rest.
                for( Size_T k = 1; k < K; ++k )
                {
                    for( Size_T i = 0; i < M; ++i )
                    {
                        combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,N>(
                            A(i,k), B(k), Scalar::One<C_T>, C(i)
                        );
                    }
                }
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors


