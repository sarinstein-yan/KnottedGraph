#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        
        template<Size_T M, Size_T N, Size_T K, AddTo_T addto, typename Scal>
        void fixed_dot_mm_clang( cptr<Scal> A, cptr<Scal> B, mptr<Scal> C )
        {
            static_assert( mat_enabledQ && (SameQ<Scal,double> || SameQ<Scal,float>), "Chosen scalar type is illegal for clang matrix extension." );
    
            // Caution: mat_T is column-major!
            // So we have to compute C^T = B^T * A^T.
            
            mat_T<K,M,Scal> AT = __builtin_matrix_column_major_load( A, K, M, K );
            mat_T<N,K,Scal> BT = __builtin_matrix_column_major_load( B, N, K, N );
            
            
            if constexpr ( addto == AddTo_T::True )
            {
                mat_T<N,M,Scal> CT = __builtin_matrix_column_major_load( C, N, M, N ) + BT * AT;
                
                __builtin_matrix_column_major_store( CT, C, N );
            }
            else
            {
                mat_T<N,M,Scal> CT =  BT * AT;
                
                __builtin_matrix_column_major_store( CT, C, N );
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors

