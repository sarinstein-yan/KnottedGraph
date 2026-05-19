#pragma once

namespace Tensors
{
    namespace Dense
    {
        template<int n, typename Scal, IntQ Int>
        class LUDecomposition final
        {
        protected:
            
            static constexpr Int max_n = n;
            
            Scal LU [max_n][max_n] = {{}};
            Int  p  [max_n]        = {};
            
            Scal buffer [max_n]    = {};
            
            
        public:
            
            // No default constructor
            LUDecomposition() = default;
            // Destructor
            ~LUDecomposition() = default;
            // Copy constructor
            LUDecomposition( const LUDecomposition & other ) = default;
            // Copy assignment operator
            LUDecomposition & operator=( const LUDecomposition & other ) = default;
            // Move constructor
            LUDecomposition( LUDecomposition && other ) = default;
            // Move assignment operator
            LUDecomposition & operator=( LUDecomposition && other ) = default;
            
            template<typename T>
            LUDecomposition( cptr<T> A_, const Int ldA )
            {
                Factorize( A_, ldA );
            }
            
            template<typename T>
            void Solve(
               cptr<T> B, const Int ldB,
               mptr<T> X, const Int ldX,
               const Int nrhs
           )
            {
                const Int max_rhs = 4 * ( (nrhs + Int(3))/ Int(4) );
                
                switch ( max_rhs )
                {
                    case 0:
                    {
                        return;
                    }
                    case 4:
                    {
                        solve<4>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 8:
                    {
                        solve<8>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 12:
                    {
                        solve<12>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 16:
                    {
                        solve<16>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 20:
                    {
                        solve<20>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 24:
                    {
                        solve<24>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 28:
                    {
                        solve<28>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    case 32:
                    {
                        solve<32>( B, ldB, X, ldX, nrhs );
                        return;
                    }
                    default:
                    {
                        solve_gen( B, ldB, X, ldX, nrhs );
                        return;
                    }
                }
            }
            
            //        template<typename T>
            //        void Solve(
            //            cptr<T> B, const Int ldB,
            //            mptr<T> X, const Int ldX,
            //            const Int nrhs
            //        )
            //        {
            //            switch ( nrhs )
            //            {
            //                case 1:
            //                {
            //                    solve<1>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 2:
            //                {
            //                    solve<2>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 3:
            //                {
            //                    solve<3>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 4:
            //                {
            //                    solve<4>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 5:
            //                {
            //                    solve<5>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 6:
            //                {
            //                    solve<6>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 7:
            //                {
            //                    solve<7>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 8:
            //                {
            //                    solve<8>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 9:
            //                {
            //                    solve<12>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 10:
            //                {
            //                    solve<12>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 11:
            //                {
            //                    solve<12>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 12:
            //                {
            //                    solve<12>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 13:
            //                {
            //                    solve<16>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 14:
            //                {
            //                    solve<16>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 15:
            //                {
            //                    solve<16>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 16:
            //                {
            //                    solve<16>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 17:
            //                {
            //                    solve<20>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 18:
            //                {
            //                    solve<20>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 19:
            //                {
            //                    solve<20>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 20:
            //                {
            //                    solve<20>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 21:
            //                {
            //                    solve<24>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 22:
            //                {
            //                    solve<24>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 23:
            //                {
            //                    solve<24>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 24:
            //                {
            //                    solve<24>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 25:
            //                {
            //                    solve<28>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 26:
            //                {
            //                    solve<28>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 27:
            //                {
            //                    solve<28>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 28:
            //                {
            //                    solve<28>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 29:
            //                {
            //                    solve<32>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 30:
            //                {
            //                    solve<32>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 31:
            //                {
            //                    solve<32>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                case 32:
            //                {
            //                    solve<32>( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //                default:
            //                {
            //                    solve_gen( B, ldB, X, ldX, nrhs );
            //                    return;
            //                }
            //            }
            //        }
            
            template<Int max_rhs, typename T>
            void solve(
                       cptr<T> B, const Int ldB,
                       mptr<T> X, const Int ldX,
                       const Int nrhs
                       )
            {
                //Goal is to solve (L U) X = B
                
                Scal Y[n][max_rhs] = {{}};
                
                // Step 1: Permute B and store it in Y.
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &B[p[i]*ldB], &Y[i][0], nrhs );
                }
                
                // Step 2: Inplace solve L Y = B
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int j = 0; j < i; ++j )
                    {
                        TOOLS_LOOP_UNROLL_FULL
                        for( Int k = 0; k < max_rhs; ++k )
                        {
                            Y[i][k] -= LU[i][j] * Y[j][k];
                        }
                    }
                }
                
                // Step 3: Inplace solve U X = Y.
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = n; i --> Int(0); )
                {
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int j = i + Int(1); j < n; ++j )
                    {
                        TOOLS_LOOP_UNROLL_FULL
                        for( Int k = 0; k < max_rhs; ++k )
                        {
                            Y[i][k] -= LU[i][j] * Y[j][k];
                        }
                    }
                    
                    const Scal U_ii_inv = Inv( LU[i][i] );
                    
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int k = 0; k < max_rhs; ++k )
                    {
                        Y[i][k] *= U_ii_inv;
                    }
                }
                
                // Step 4: Write result.
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &Y[i][0],  &X[i*ldX], nrhs);
                }
            }
            
            
            template<typename T>
            void solve_gen(
                cptr<T> B, const Int ldB,
                mptr<T> X, const Int ldX,
                const Int nrhs
            )
            {
                //Goal is to solve (L U) X = B
                
                Tiny::VectorList<n,Scal,Int> Y (nrhs);
                
                // Step 1: Permute B and store it in Y.
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &B[p[i]*ldB], &Y[i][0], nrhs );
                }
                
                // Step 2: Inplace solve L Y = B
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int j = 0; j < i; ++j )
                    {
                        for( Int k = 0; k < nrhs; ++k )
                        {
                            Y[i][k] -= LU[i][j] * Y[j][k];
                        }
                    }
                }
                
                // Step 3: Inplace solve U X = Y.
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = n; i --> Int(0); )
                {
                    for( Int j = i + Int(1); j < n; ++j )
                    {
                        for( Int k = 0; k < nrhs; ++k )
                        {
                            Y[i][k] -= LU[i][j] * Y[j][k];
                        }
                    }
                    
                    const Scal U_ii_inv = Inv( LU[i][i] );
                    
                    for( Int k = 0; k < nrhs; ++k )
                    {
                        Y[i][k] *= U_ii_inv;
                    }
                }
                
                // Step 4: Write result.
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &Y[i][0], &X[i*ldX], nrhs );
                }
                
            } // Solve_gen
            
            
            template<typename T>
            void Factorize( cptr<T> A_, const Int ldA )
            {
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &A_[ldA*i], &LU[i][0], n );
                }
                
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    p[i] = i;
                }
                
                TOOLS_LOOP_UNROLL_FULL
                for( Int k = 0; k < n-1; ++k )
                {
                    // Find pivot index r = argmax |A[i][k]|, i = k,...,n.
                    Int r = k;
                    
                    Scal a = Abs(LU[k][k]);
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int i = k+1; i < n; ++i )
                    {
                        const Scal b = Abs(LU[i][k]);
                        
                        if( b > a )
                        {
                            a = b;
                            r = i;
                        }
                    }
                    
                    // Swap rows;
                    if( r != k )
                    {
                        std::swap( p[r], p[k] );
                        std::swap_ranges( &LU[r][0], &LU[r+1][0], &LU[k][0] );
                    }
                    
                    const Scal LU_kk_inv = Inv( LU[k][k] );
                    
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int i = k+1; i < n; ++i )
                    {
                        LU[i][k] *= LU_kk_inv;
                        
                        TOOLS_LOOP_UNROLL_FULL
                        for( Int j = k+1; j < n; ++j )
                        {
                            LU[i][j] -= LU[i][k] * LU[k][j];
                        }
                    }
                }
            }
            
            template<typename T>
            void WriteFactors( mptr<T> A_, const Int ldA ) const
            {
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &LU[i][0], &A_[ldA*i], n );
                }
            }
            
            template<typename I>
            void WritePermutation( mptr<I> p_ ) const
            {
                copy_buffer( &p[0], p_, n );
            }
            
            template<typename T>
            void ReadFactors( cptr<T> A_, const Int ldA )
            {
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &A_[ldA*i], &LU[i][0], n );
                }
            }
            
            template<typename I>
            void ReadPermutation( cptr<I> const p_ )
            {
                copy_buffer( p_, &p[0], n );
            }
            
        }; // class LUDecomposition
        
    } // namespace Dense
    
} // namespace Tensors
