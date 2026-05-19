#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        
        template<Size_T M, Size_T N, Size_T K, AddTo_T addto, 
            typename A_T, typename B_T, typename C_T
        >
        void fixed_dot_mm( cptr<A_T> A, cptr<B_T> B, mptr<C_T> C )
        {
            // A is of size M X K
            // B is of size K X N
            // C is of size M X N
            
            // All matrices are assumed to be in row-major storage.
        
            constexpr bool all_doubles = SameQ<A_T,double> && SameQ<B_T,double> && SameQ<C_T,double>;
            
            constexpr bool all_floats  = SameQ<A_T,float > && SameQ<B_T,float > && SameQ<C_T,float >;
            
            
            if constexpr( mat_enabledQ && ( all_doubles || all_floats ) )
            {
                fixed_dot_mm_clang<M,N,K,addto>( A, B, C );
            }
            else
            {
//                fixed_dot_mm_vec  <M,N,K,addto>( A, B, C );
                
                fixed_dot_mm_naive<M,N,K,addto>( A, B, C );
            }
        }
        
                
    } // namespace Tiny
    
} // namespace Tensors

