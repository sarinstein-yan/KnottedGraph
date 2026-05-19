#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
        template<int n_, typename Scal_, IntQ Int_>
        class SelfAdjointTridiagonalMatrix final
        {
            // Uses only upper triangle.
            
            // TODO: Add value semantics?
            
        public:
            
            using Class_T = SelfAdjointTridiagonalMatrix;
            
#include "Tiny_Details.hpp"
            
        public:
            
            static constexpr Int n = n_;
            
            using Upper_T  = Vector<n-1,Scal,Int>;
            using Vector_T = Vector<n,  Scal,Int>;
            using Matrix_T = Matrix<n,n,Real,Int>;
            
        protected:
            
            Vector_T diag;  //the main diagonal (should actually only have real values on it.
            Upper_T  upper; //upper diagonal
            
            
            Matrix_T Q;
            
        public:
            
///######################################################
///##                     Memory                       ##
///######################################################
            
            
            SelfAdjointTridiagonalMatrix() = default;
            
            ~SelfAdjointTridiagonalMatrix() = default;
            
            explicit SelfAdjointTridiagonalMatrix( cref<Scal> init )
            :   diag  { init }
            ,   upper { init }
            {}
            
            
            TOOLS_FORCE_INLINE void SetZero()
            {
                zerofy_buffer<n  >( &diag[0]  );
                zerofy_buffer<n-1>( &upper[0] );
            }
            
///######################################################
///##                     Access                       ##
///######################################################
            
            
            TOOLS_FORCE_INLINE mref<Real> Diag( const Int i )
            {
                return diag[i];
            }
            
            TOOLS_FORCE_INLINE cref<Real> Diag( const Int i ) const
            {
                return diag[i];
            }
            
            TOOLS_FORCE_INLINE mref<Scal> Upper( const Int i )
            {
                return upper[i];
            }
            
            TOOLS_FORCE_INLINE cref<Scal> Upper( const Int i ) const
            {
                return upper[i];
            }
            
            TOOLS_FORCE_INLINE Scal Lower( const Int i )
            {
                return Conj(upper[i]);
            }
            
            
            TOOLS_FORCE_INLINE friend SelfAdjointTridiagonalMatrix operator+( const SelfAdjointTridiagonalMatrix & x, const SelfAdjointTridiagonalMatrix & y )
            {
                SelfAdjointTridiagonalMatrix z;
                for( Int i = 0; i < n; ++i )
                {
                    z.diag[i] = x.diag[i] + y.diag[i];
                }
                for( Int i = 0; i < n-1; ++i )
                {
                    z.upper[i] = x.upper[i] + y.upper[i];
                }
                
                return z;
            }
            
            TOOLS_FORCE_INLINE mref<SelfAdjointTridiagonalMatrix> operator+=( const SelfAdjointTridiagonalMatrix & B )
            {
                add_to_buffer<n>  ( &B.diag[0],  &diag[0]       );
                add_to_buffer<n-1>( &B.upper[0], &diag.upper[0] );
            }
            
            TOOLS_FORCE_INLINE void Dot( const Vector_T & x, Vector_T & y ) const
            {
                if constexpr ( n >= Int(1) )
                {
                    y[0] = diag[0] * x[0];
                }
                else if constexpr ( n > Int(1) )
                {
                    y[0] = diag[0] * x[0] + upper[0] * x[1];
                }
                
                for( Int i = 1; i < n-Int(2); ++ i )
                {
                    y[i] = upper[i-1] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1];
                }
                
                if constexpr ( n >= Int(2) )
                {
                    y[n-1] = diag[n-1] * x[n-1];
                }
                else if constexpr  ( n > Int(2) )
                {
                    y[n-1] = upper[n-2] * x[n-2] + diag[n-1] * x[n-1];
                }
            }
            
            [[nodiscard]] std::string friend ToString( cref<SelfAdjointTridiagonalMatrix> M )
            {
                std::string s;
                
                s += "{\n";
                s += "\tdiag  = ";
                s += OutString::FromVector(&M.diag[0],n);
                s += "\n\tupper = ";
                s += OutString::FromVector(&M.upper[0],n-1);
                s += "\n}";
                
                return s;
            }
            
            template<typename T = Scal>
            void ToMatrix( Matrix<n,n,T,Int> & B ) const
            {
                if( n <= 0 )
                {
                    return;
                }
                
                B.SetZero();
                
                for( Int i = 0; i < n-1; ++i )
                {
                    B[i  ][i  ] = static_cast<T>(diag[i]);
                    B[i  ][i+1] = static_cast<T>(upper[i]);
                    B[i+1][i  ] = static_cast<T>(Conj(upper[i]));
                }
                B[n-1][n-1] = static_cast<T>(diag[n-1]);
            }
            
        public:
            
            template<typename S>
            TOOLS_FORCE_INLINE std::enable_if_t<Scalar::RealQ<Scal>,void >
            QRAlgorithm(
                mref<Matrix<n,n,S,Int>> & Q_result,
                mref<Vector<n,  S,Int>> & eigs,
                const Real tol = eps_sqrt,
                const Int max_iter = 16
            )
            {
                // Computes a unitary matrix Q and and a vector eigs such that
                // Q . A . Q^T = diag(eigs).
                // Caution: Performs computations in-place, so it modifies A.
                
                
                if constexpr ( n <= 0 )
                {
                    return;
                }
                else if constexpr ( n == 1 )
                {
                    eigs[0] = diag[0];
                    Q_result[0][0] = Scalar::One<Scal>;
                }
                else if constexpr ( n == 2 )
                {
                    Q.SetIdentity();
                    
                    qr_iteration_2x2<true>( 0 );
                    
                    eigs[0] = diag[0];
                    eigs[1] = diag[1];
                    
                    Q.Write( Q_result.data() );
                }
                else if constexpr ( n >= 3 )
                {
                    Q.SetIdentity();
                    
                    qr_find_blocks<true>( 0, n, tol, max_iter, true );
                    
                    diag.Write( eigs.data() );
                    Q.Write( Q_result.data() );
                }
            }
            
            template<typename S>
            TOOLS_FORCE_INLINE std::enable_if_t<Scalar::RealQ<Scal>, void>
            QRAlgorithm(
                mref<Vector<n,S,Int>> & eigs,
                const Real tol = eps_sqrt,
                const Int max_iter = 16
            )
            {
                // Computes the eigenvalues only.
                // Caution: Performs computations in-place, so it modifies A.
                
                if constexpr ( n <= 0 )
                {
                    return;
                }
                else if constexpr ( n == 1 )
                {
                    eigs[0] = diag[0];
                }
                else if constexpr ( n == 2 )
                {
                    qr_iteration_2x2<false>( 0 );
                    
                    eigs[0] = diag[0];
                    eigs[1] = diag[1];
                }
                else if constexpr ( n >= 3 )
                {
                    qr_find_blocks<false>( 0, n, tol, max_iter, true );
                    
                    diag.Write( eigs.data() );
                }
            }
            
        public:
            
            template<bool track_rotationsQ = true>
            TOOLS_FORCE_INLINE std::enable_if_t<Scalar::RealQ<Scal>, void >
            qr_find_blocks( 
                const Int begin, const Int end, const Real tol, const Int max_iter,
                const bool first_call
            )
            {
                if( end == begin + 1)
                {
                    return;
                }
                
                Int a = begin;
                
                while( a < end - 1 )
                {
                    // Find the first a with upper[a] != 0.
                    
                    while( (a < end - 1 ) && ( Abs(upper[a]) < tol * ( Abs(diag[a]) + Abs(diag[a+1]) ) ) )
                    {
                        ++a;
                    }
                    
                    Int b = Tools::Min( a + 1, end );
                    
                    // Find the first b with upper[b] == 0.
                    while( (b < end ) && ( Abs(upper[b-1]) >= tol * ( Abs(diag[b-1]) + Abs(diag[b]) ) ) )
                    {
                        ++b;
                    }
                    
                    if( ( !first_call) && (a == begin) && (b == end) )
                    {
                        wprint( ClassName() +"::qr_find_blocks was not able to deflate the current block. Skipping this block. Expect errors in the eigensystem." );

                        TOOLS_DUMP(diag);
                        TOOLS_DUMP(upper);
                        TOOLS_DUMP(a);
                        TOOLS_DUMP(b);
                        
                        return;
                    }
                    
                    if ( b == a + 2 )
                    {
                        qr_iteration_2x2<track_rotationsQ>( a );
                    }
                    else if ( b > a + 2 )
                    {
                        qr_iteration<track_rotationsQ>( a, b, tol, 0, max_iter );
                    }
                    
                    a = b;
                }
            }
            
            
            template<bool track_rotationsQ = true>
            TOOLS_FORCE_INLINE std::enable_if_t< Scalar::RealQ<Scal>, void >
            qr_iteration_2x2( const Int k )
            {
                const Real a_0 = diag [k  ];
                const Real a_1 = diag [k+1];
                const Real b_0 = upper[k  ];
                
                /*
                 * Find c, s such that
                 *
                 *    /         \  /            \  /         \
                 *    |  c  -s  |  |  a_0  b_0  |  |   c  s  |
                 *    |  s   c  |  |  b_0  a_1  |  |  -s  c  |
                 *    \         /  \            /  \         /
                 *
                 *  is diagonal.
                */
                
                if( Abs(b_0) > eps * ( Abs(a_0) + Abs(a_1) ) )
                {
                    const Real d = a_0 - a_1;
                    const Real delta = Sqrt( d * d + four * b_0 * b_0 );
                    
                    const Real s_ = (d + delta) / (two * b_0);
                    const Real factor = InvSqrt(one + s_ * s_);
                    
                    const Real c = factor;
                    const Real s = s_ * factor;
                    
                    diag [k  ] = half * (a_0 + a_1 - delta);
                    diag [k+1] = half * (a_0 + a_1 + delta);
                    upper[k  ] = 0;
                    
                    if constexpr ( track_rotationsQ )
                    {
                        // Multiply Q by Givens(c,s,k,k+1) from the right.
                        Q.GivensRight(c,s,k,k+1);
                    }
                }
            }
            
            template<bool track_rotationsQ = true>
            TOOLS_FORCE_INLINE std::enable_if_t<Scalar::RealQ<Scal>, void >
            qr_iteration( 
                const Int begin, const Int end, const Real tol, const Int iter, const Int max_iter
            )
            {
                // Performs the implicit QR algorithm in the block A[begin..end-1][begin..end-1].
                // Assumes that end > begin + 2 (otherwise we would have called qr_algorithm_2x2(begin) or stopped.
                
                // Compute Wilkinson’s shift mu
                const Real a = diag [end-1];
                const Real b = upper[end-2];

                const Real d_ = half * (diag[end-2] - a);

                const Real mu = ( Abs(d_) <= eps * ( Abs(diag[end-2]) + Abs(a) ) )
                    ?
                    a - Abs(b)
                    :
                    a - b * b / ( d_ + (d_ >= zero ? one : -one ) * Sqrt( d_ * d_ + b * b ) );
                
                // Implicit QR step begins here
                
                Real x = diag [begin] - mu;
                Real y = upper[begin];
                
                {
                    const Int k = begin;

                    /*
                     * Find c, s such that
                     *
                     *    /         \  /     \   /       \
                     *    |  c  -s  |  |  x  | = |  rho  |
                     *    |  s   c  |  |  y  |   |   0   |
                     *    \         /  \     /   \       /
                     *
                     */
                    
                    const Real rho = Sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[k] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real d = diag[k] - diag[k+1];
                        const Real z = ( two * c * upper[k] + d * s ) * s;
                        diag [k  ] -= z;
                        diag [k+1] += z;
                        upper[k  ]  = d * c * s + (c * c - s * s) * upper[k];
                        x = upper[k];
                        y = - s * upper[k+1];
                        upper[k+1] *= c;
                        
                        if constexpr ( track_rotationsQ )
                        {
                            // Multiply Q by Givens(c,s,k,k+1) from the right.
                            Q.GivensRight(c,s,k,k+1);
                        }
                    }
                }
                
                for( Int k = begin + 1; k < end-2; ++k )
                {
                    /*
                     * Find c, s such that
                     *
                     *    /         \  /     \   /       \
                     *    |  c  -s  |  |  x  | = |  rho  |
                     *    |  s   c  |  |  y  |   |   0   |
                     *    \         /  \     /   \       /
                     *
                     */
                    
                    const Real rho = Sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[k] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real w = c * x - s * y;
                        const Real d = diag[k] - diag[k+1];
                        const Real z = ( two * c * upper[k] + d * s ) * s;
                        diag[k  ] -= z;
                        diag[k+1] += z;
                        upper[k] = d * c * s + (c * c - s * s) * upper[k];
                        x = upper[k];
                        upper[k-1] = w;
                        y = - s * upper[k+1];
                        upper[k+1] *= c;
                        
                        if constexpr ( track_rotationsQ )
                        {
                            // Multiply Q by Givens(c,s,k,k+1) from the right.
                            Q.GivensRight(c,s,k,k+1);
                        }
                    }
                }
                
                {
                    const Int k = end - 2;
                    
                    /*
                     * Find c, s such that
                     *
                     *    /         \  /     \   /       \
                     *    |  c  -s  |  |  x  | = |  rho  |
                     *    |  s   c  |  |  y  |   |   0   |
                     *    \         /  \     /   \       /
                     */
                    
                    const Real rho = Sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[k] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real w = c * x - s * y;
                        const Real d = diag[k] - diag[k+1];
                        const Real z = ( two * c * upper[k] + d * s ) * s;
                        diag [k  ] -= z;
                        diag [k+1] += z;
                        upper[k  ]  = d * c * s + (c * c - s * s) * upper[k];
                        x = upper[k];
                        upper[k-1] = w;
                        
                        if constexpr ( track_rotationsQ )
                        {
                            // Multiply Q by Givens(c,s,k,k+1) from the right.
                            Q.GivensRight(c,s,k,k+1);
                        }
                    }
                }
                // Implicit QR step ends here
                
                
                
                // Check for possible deflation and do the next iteration.
                
                if( Abs(upper[end-2]) > tol * ( Abs(diag[end-2]) + Abs(diag[end-1]) ) )
                {
                    if( iter < max_iter )
                    {
                        qr_iteration<track_rotationsQ>( begin, end, tol, iter+1, max_iter );
                    }
                    else
                    {
                        qr_find_blocks<track_rotationsQ>( begin, end, tol, max_iter, false );
                    }
                }
                else
                {
                    if ( end == begin + 3 )
                    {
                        qr_iteration_2x2<track_rotationsQ>( begin );
                    }
                    else
                    {
                        qr_find_blocks<track_rotationsQ>( begin, end-1, tol, max_iter, true );
                    }
                }
            }
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return ct_string("Tiny::SelfAdjointTridiagonalMatrix")
                    + "<" + to_ct_string(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + ">";
            }
            
        };
        
    } // namespace Tiny
    
} // namespace Tensors
