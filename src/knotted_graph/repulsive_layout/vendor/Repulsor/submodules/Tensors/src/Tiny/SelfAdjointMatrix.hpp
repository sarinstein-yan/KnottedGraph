#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template<int n_, typename Scal_, IntQ Int_>
        class SelfAdjointMatrix final
        {
            /// This class uses only upper triangle.
            
        public:
            
            using Class_T = SelfAdjointMatrix;
            
#include "Tiny_Details.hpp"

        public:
            
            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scal,Int>;
            
        protected:
            
//            std::array<std::array<Scal,n>,n> A;
            
            Scal A[n][n];
            
            // TODO: Needs swap routine.
//            Scal A [n][n];
            
        public:
            
            SelfAdjointMatrix() = default;

            ~SelfAdjointMatrix() = default;

            SelfAdjointMatrix(std::nullptr_t) = delete;

            explicit SelfAdjointMatrix( const Scal * a )
            {
                Read(a);
            }

            // Copy constructor
            explicit SelfAdjointMatrix( const Class_T & other )
            {
                Read( &other.A[0][0] );
            }

            // Copy assignment operator
            mref<SelfAdjointMatrix> operator=( const SelfAdjointMatrix other )
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                swap(*this, other);

                return *this;
            }

            /* Move constructor */
            SelfAdjointMatrix( SelfAdjointMatrix && other ) noexcept
            {
                swap(*this, other);
            }

            explicit SelfAdjointMatrix( cref<Scal> init )
            {
                Fill(init);
            }


#include "Tiny_UpperTriangular_Common.hpp"
            
///######################################################
///##                  Arithmetic                      ##
///######################################################
            
        public:
            
            template<class T>
            TOOLS_FORCE_INLINE mref<SelfAdjointMatrix> operator+=( cref<SelfAdjointMatrix<n,T,Int>> B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
                return *this;
            }
            
            template<class T>
            TOOLS_FORCE_INLINE mref<SelfAdjointMatrix> operator-=( cref<SelfAdjointMatrix<n,T,Int>> B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] -= B.A[i][j];
                    }
                }
                return *this;
            }
            
            template<class T>
            TOOLS_FORCE_INLINE mref<SelfAdjointMatrix> operator*=( cref<SelfAdjointMatrix<n,T,Int>> B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
                return *this;
            }
            
            template<class T>
            TOOLS_FORCE_INLINE mref<SelfAdjointMatrix> operator/=( cref<SelfAdjointMatrix<n,T,Int>> B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] /= B.A[i][j];
                    }
                }
                return *this;
            }

            template<Op op>
            void LowerFromUpper() const
            {}

            template< AddTo_T addto >
            TOOLS_FORCE_INLINE friend void Dot( const SelfAdjointMatrix & M, const Vector_T & x, Vector_T & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    Scal y_i ( (addto == Overwrite) ? Scalar::Zero<Scal> : y[i] );
                    
                    for( Int j = 0; j < i; ++j )
                    {
                        y_i += Conj(M[j][i]) * x[j];
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        y_i += M[i][j] * x[j];
                    }
                    
                    y[i] = y_i;
                }
            }
            
            TOOLS_FORCE_INLINE friend Scal InnerProduct( const SelfAdjointMatrix & G, const Vector_T & x, const Vector_T & y )
            {
                Scal result (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    Scal z_i (0);
                    
                    for( Int j = 0; j < i; ++j )
                    {
                        z_i += G.A[j][i] * x.v[j];
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        z_i += Conj(G.A[i][j]) * x.v[j];
                    }

                    result += Conj(y.v[i]) * z_i;
                }

                return result;
            }
            
            void Cholesky()
            {
                // In-place Cholesky factorization.
                for( Int k = 0; k < n; ++k )
                {
                    const Real a { Sqrt( Abs(A[k][k]) ) };
                    
                    A[k][k] = a;
                    
                    const Real ainv { Inv(a) };
                    
                    for( Int j = k+1; j < n; ++j )
                    {
                        A[k][j] *= ainv;
                    }
                    
                    for( Int i = k+1; i < n; ++i )
                    {
                        for( Int j = i; j < n; ++j )
                        {
                            A[i][j] -= Conj(A[k][i]) * A[k][j];
                        }
                    }
                }
            }
            
            void Cholesky( UpperTriangularMatrix<n,Scal,Int> & U ) const
            {
                // Computes and returns the upper factor U = L ^H such that A = U^H * U.
                
                Write( U.data() );
                
                for( Int k = 0; k < n; ++k ) // for each row
                {
                    const Real u { Sqrt( Abs(U[k][k]) ) };
                    
                    U[k][k] = u;
                    
                    const Real uinv { Inv(u) };
                    
                    // scale_buffer( uinv, &U[k][k+1], n-k-1 );
                    for( Int j = k+1; j < n; ++j )
                    {
                        U[k][j] *= uinv;
                    }
                    
                    
                    for( Int i = k+1; i < n; ++i ) // for each row i below k
                    {
                        for( Int j = i; j < n; ++j )
                        {
                            U[i][j] -= Conj(U[k][i]) * U[k][j];
                        }
                    }
                }
            }
            
            
            void CholeskySolve( const Vector_T & b, Vector_T & x ) const
            {
                x = b;
                CholeskySolve(x);
            }
            
            void CholeskySolve( Vector_T & x ) const
            {
                //In-place solve.
                
                // Lower triangular back substitution
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < i; ++j )
                    {
                        x[i] -= Conj(A[j][i]) * x[j];
                    }
                    x[i] /= A[i][i];
                }
                
                // Upper triangular back substitution
                for( Int i = n; i --> Int(0); )
                {
                    for( Int j = i + Int(1); j < n; ++j )
                    {
                        x[i] -= A[i][j] * x[j];
                    }
                    x[i] /= A[i][i];
                }
            }
            
            template< int K >
            void CholeskySolve(Tiny::Matrix<n,K,Scal,Int> & X) const
            {
                //In-place solve.
                
                // Lower triangular back substitution
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < i; ++j )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            const Scal bar_a_ji = Conj(A[j][i]);
                            
                            X[i][k] -= bar_a_ji * X[j][k];
                        }
                    }
                    
                    const Scal A_ii_inv = Inv(A[i][i]);
                    
                    for( Int k = 0; k < K; ++k )
                    {
                        X[i][k] *= A_ii_inv;
                    }
                }
                
                // Upper triangular back substitution
                for( Int i = n; i --> Int(0); )
                {
                    for( Int j = i + Int(1); j < n; ++j )
                    {
                        const Scal A_ij = A[i][j];
                        
                        for( Int k = 0; k < K; ++k )
                        {
                            X[i][k] -= A_ij * X[j][k];
                        }
                    }
                    
                    const Scal A_ii_inv = Inv(A[i][i]);
                    
                    for( Int k = 0; k < K; ++k )
                    {
                        X[i][k] *= A_ii_inv;
                    }
                }
            }
            
            Real SmallestEigenvalue( const Real tol = eps, const Int max_iter = 16 ) const
            {
                if constexpr ( n == 1 )
                {
                    return A[0][0];
                }
                
                if constexpr ( n == 2 )
                {
                    Real diag[2] = { Re(A[0][0]), Re(A[1][1]) };
                    
                    Real lambda_min = half * (
                        diag[0] + diag[1] - Sqrt( AbsSquared( diag[0]-diag[1] ) + four * AbsSquared( A[0][1] ) )
                    );
                    
                    return lambda_min;
                }
                
                if constexpr ( n == 3 )
                {
                    constexpr Real Pi_Third = Scalar::Pi<Real> * Scalar::Third<Real>;
                    constexpr Real Pi_Two_Third = two * Pi_Third;
                    
                    Real lambda_min;
                    
                    Real diag [3] = { Re(A[0][0]), Re(A[1][1]), Re(A[2][2]) };
                    
                    const Real p1 = AbsSquared(A[0][1]) + AbsSquared(A[0][2]) + AbsSquared(A[1][2]);
                    
                    if( p1 < eps_squared * ( AbsSquared(diag[0]) + AbsSquared(diag[1]) + AbsSquared(diag[2])) )
                    {
                        // A is diagonal
                        lambda_min = Min( diag[0], Min( diag[1], diag[2] ) );
                    }
                    else
                    {
                        const Real q         ( ( diag[0] + diag[1] + diag[2] ) * Scalar::Third<Real> );
                        const Real delta [3] { diag[0]-q, diag[1]-q, diag[2]-q } ;
                        const Real p2        ( AbsSquared(delta[0]) + AbsSquared(delta[1]) + AbsSquared(delta[2]) + two * p1 );
                        const Real p         ( Sqrt( p2 * Scalar::Sixth<Real> ) );
                        const Real pinv ( Inv(p) );
                        const Real b11  ( delta[0] * pinv );
                        const Real b22  ( delta[1] * pinv );
                        const Real b33  ( delta[2] * pinv );
                        const Scal b12  (  A[0][1] * pinv );
                        const Scal b13  (  A[0][2] * pinv );
                        const Scal b23  (  A[1][2] * pinv );
                        
                        const Real r = half * (two * Re( b12 * b23 * Conj(b13) ) - AbsSquared(b23) * b11 - AbsSquared(b13) * b22 - AbsSquared(b12) * b33 + b11 * b22 * b33 );
                        
                        
                        const Real phi = ( r <= - one ) ? Pi_Third : ( ( r >= one ) ? zero : acos(r) * Scalar::Third<Real> );
                        
                        
                        // The eigenvalues are ordered this way: eig2 <= eig1 <= eig0.
                        
                        //                    Scal eig0 ( q + two * p * cos( phi ) );
                        //                    Scal eig2 ( q + two * p * cos( phi + Pi_Two_Third ) );
                        //                    Scal eig1 ( three * q - eig0 - eig2 );
                        
                        lambda_min = Re( q + two * p * cos( phi + Pi_Two_Third ) );
                    }
                    
                    return lambda_min;
                }
                
                if constexpr ( n > Int(3) )
                {
                    Vector<n,Real,Int> v;
    
                    Eigenvalues(v, tol, max_iter );
                    
                    return v.Min();
                }
                
                return 0;
            }
            
            void HessenbergDecomposition(
                Matrix                      <n,n,Scal,Int> & U,
                SelfAdjointTridiagonalMatrix<n,  Real,Int> & T
            ) const
            {
                // Computes a unitary matrix U and and a self-adjoint tridiagonal matrix T such that U . T . U^H = A.

                if constexpr ( n == 1 )
                {
                    T.Diag(0) = Re(A[0][0]);
                    U[0][0]   = one;
                }
                
                if constexpr ( n == 2 )
                {
                    T.Diag(0)  = Re(A[0][0]);
                    T.Diag(1)  = Re(A[1][1]);
                    T.Upper(0) = Abs(A[0][1]);
                    U[0][0] = 1;
                    U[0][1] = 0;
                    U[1][0] = 0;
                    U[1][1] = (T.Upper(0) == zero) ? one : Conj(A[0][1]) / T.Upper(0);
                }
                
                if constexpr ( n > Int(2) )
                {
                    Matrix<n,n, Scal, Int> B ;
                    Write( &B[0][0] );
                    
//                    Scal u [n-2][n]; // vectors of the Householder reflections.
//                    Scal v [n];      // some scratch space
                    
                    Vector_T u [n - Int(2)]; // vectors of the Householder reflections.
                    Vector_T v      ; // some scratch space
                    
                    for( Int k = 0; k < n - Int(2); ++k )
                    {
                        // Compute k-th Householder reflection vector.
                        
//                        u[k][k] = 0; // We know that u[k][0] = ... = u[k][k] = 0; we just use this implicitly!
//                        for( Int i = 0; i < k+1; ++i )
//                        {
//                            u[k][i] = 0
//                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] = Conj(B[k][i]);
                        }
                        
                        Real uu = 0;
                        for( Int i = k+1; i < n; ++i )
                        {
                            uu += AbsSquared(u[k][i]);
                        }
                        
                        if( uu < eps_squared )
                        {
                            // TODO: One could actually use this to split the matrix into two.
                            continue;
                        }
                        
                        Real u_norm = Sqrt( uu );
                        
                        Scal u_pivot (u[k][k+1]);
                        
                        Real abs_squared_u_pivot = AbsSquared(u_pivot);
                        
                        const Scal rho (
                            (
                                Scalar::ComplexQ<Scal>
                                ?
                                ( abs_squared_u_pivot <= eps_squared * uu ) ? one : -u_pivot / Sqrt(abs_squared_u_pivot)
                                :
                                ( u_pivot > zero ) ? -one : one
                            )
                        );
                        
                        uu -= abs_squared_u_pivot;
                        
                        u[k][k+1] -= rho * u_norm;
                        
                        uu += AbsSquared(u[k][k+1]);
                        
                        Real u_norm_inv ( InvSqrt( uu ) );
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] *= u_norm_inv;
                        }
                        
                        
                        // Compute v_k = B . u_k and  u_k . B . u_k
                        
                        Scal ubarBu_accumulator (0);
                        
                        {
                            const Int i = k;
                            
                            Scal Bu_i (0);
                            
                            // We can skip this.
//                            for( Int j = k+1; j < i; ++j )
//                            {
//                                Bu_i += Conj(B[j][i]) * u[k][j];
//                            }
                            
                            for( Int j = k+1; j < n; ++j ) // we implicitly use u[k][k] == 0
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            // We implicitly use u[k][k] = 0;
//                            ubarBu += Conj(u[k][i]) * Bu_i;
                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            Scal Bu_i (0);
                            
                            for( Int j = k+1; j < i; ++j )
                            {
                                Bu_i += Conj(B[j][i]) * u[k][j];
                            }
                            
                            for( Int j = i; j < n; ++j )
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            ubarBu_accumulator += Conj(u[k][i]) * Bu_i;
                        }
                        
                        Real ubarBu = Re(ubarBu_accumulator);

                        {
                            const Int i = k;
                            const Scal a ( (- two) * v[k] );

                            for( Int j = i+1; j < n; ++j )  // Exploit that u[k][i] = u[k][k] == 0
                            {
                                B[k][j] += a * Conj(u[k][j]);
                            }
                        }
                        
                        // Apply Householder reflection to both sides of B.
                        for( Int i = k+1; i < n; ++i )
                        {
                            const Scal a ( (four * ubarBu) * u[k][i] - two * v[i]);
                            const Scal b ( two * u[k][i] );
                            
                            for( Int j = i; j < n; ++j )
                            {
                                B[i][j] += a * Conj(u[k][j]) - b * Conj(v[j]);
                            }
                        }
                    }
                                       
                    // We want a purely real tridiagonal matrix...
                    for( Int i = 0; i < n-1; ++i )
                    {
                        T.Diag(i)  = Re(B[i][i]);
                        T.Upper(i) = ( Scalar::ComplexQ<Scal>? Abs(B[i][i+1]): B[i][i+1] );
                    }
                    T.Diag(n-1)  = Re(B[n-1][n-1]);
                    
                    // ... hence we put appropriate unimodular numbers on the diagonal of U.
                    if constexpr ( Scalar::ComplexQ<Scal> )
                    {
                        U.SetZero();
                        U[0][0] = 1;
                        for( Int k = 1; k < n; ++k )
                        {
                            Real absb = T.Upper(k-1);
                            U[k][k]   = (absb == zero) ? one : U[k-1][k-1] * Conj(B[k-1][k]) / absb;
                        }
                    }
                    else
                    {
                        U.SetIdentity();
                    }
                        
                    // Apply Householder transformations from the left (reverse order to safe some flops).
                    for( Int k = n-2; k --> Int(0) ; )
                    {
                        // Compute v = Conj(u[k]) * U;
                        for( Int j = k + Int(1); j < n; ++j )
                        {
                            Scal ubarU_j = 0;
                            for( Int i = k + Int(1); i < n; ++i )
                            {
                                ubarU_j += Conj(u[k][i]) * U[i][j];
                            }
                            v[j] = ubarU_j;
                        }
                        
                        for( Int i = k + Int(1); i < n; ++i )
                        {
                            const Scal a = two * u[k][i];
                            for( Int j = k + Int(1); j < n; ++j )
                            {
                                U[i][j] -= a * v[j];
                            }
                        }
                    }
                }
            }
            
            void HessenbergDecomposition(
                SelfAdjointTridiagonalMatrix<n,Real,Int> & T
            ) const
            {
                // Computes a unitary matrix U and and a self-adjoint tridiagonal matrix T such that U . T . U^H = A.

                if constexpr ( n == 1 )
                {
                    T.Diag(0) = Re(A[0][0]);
                }
                
                if constexpr ( n == 2 )
                {
                    T.Diag(0)  = Re(A[0][0]);
                    T.Diag(1)  = Re(A[1][1]);
                    T.Upper(0) = Abs(A[0][1]);
                }
                
                if constexpr ( n > Int(2) )
                {
                    Matrix<n,n, Scal, Int> B ;
                    Write( &B[0][0] );
                    
//                    Scal u [n-2][n]; // vectors of the Householder reflections.
//                    Scal v [n];      // some scratch space
                    
                    Vector_T u [n - Int(2)]; // vectors of the Householder reflections.
                    Vector_T v      ; // some scratch space
                    
                    for( Int k = 0; k < n - Int(2); ++k )
                    {
                        // Compute k-th Householder reflection vector.
                        
//                        u[k][k] = 0; // We know that u[k][0] = ... = u[k][k] = 0; we just use this implicitly!
//                        for( Int i = 0; i < k+1; ++i )
//                        {
//                            u[k][i] = 0
//                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] = Conj(B[k][i]);
                        }
                        
                        Real uu = 0;
                        for( Int i = k+1; i < n; ++i )
                        {
                            uu += AbsSquared(u[k][i]);
                        }
                        
                        Real u_norm = Sqrt( uu );
                        
                        Scal u_pivot (u[k][k+1]);
                        
                        Real abs_squared_u_pivot = AbsSquared(u_pivot);
                        
                        const Scal rho (
                            (
                                Scalar::ComplexQ<Scal>
                                ?
                                ( abs_squared_u_pivot <= eps_squared * uu ) ? one : -u_pivot / Sqrt(abs_squared_u_pivot)
                                :
                                ( u_pivot > zero ) ? -one : one
                            )
                        );
                        
                        uu -= abs_squared_u_pivot;
                        
                        u[k][k+1] -= rho * u_norm;
                        
                        uu += AbsSquared(u[k][k+1]);
                        
                        Real u_norm_inv ( InvSqrt( uu ) );
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] *= u_norm_inv;
                        }
                        
                        
                        // Compute v_k = B . u_k and  u_k . B . u_k
                        
                        Scal ubarBu_accumulator (0);
                        
                        {
                            const Int i = k;
                            
                            Scal Bu_i (0);
                            
                            // We can skip this.
//                            for( Int j = k+1; j < i; ++j )
//                            {
//                                Bu_i += Conj(B[j][i]) * u[k][j];
//                            }
                            
                            for( Int j = k+1; j < n; ++j ) // we implicitly use u[k][k] == 0
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            // We implicitly use u[k][k] = 0;
//                            ubarBu += Conj(u[k][i]) * Bu_i;
                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            Scal Bu_i (0);
                            
                            for( Int j = k+1; j < i; ++j )
                            {
                                Bu_i += Conj(B[j][i]) * u[k][j];
                            }
                            
                            for( Int j = i; j < n; ++j )
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            ubarBu_accumulator += Conj(u[k][i]) * Bu_i;
                        }
                        
                        Real ubarBu = Re(ubarBu_accumulator);

                        {
                            const Int i = k;
                            const Scal a ( (- two) * v[k] );

                            for( Int j = i+1; j < n; ++j )  // Exploit that u[k][i] = u[k][k] == 0
                            {
                                B[k][j] += a * Conj(u[k][j]);
                            }
                        }
                        
                        // Apply Householder reflection to both sides of B.
                        for( Int i = k+1; i < n; ++i )
                        {
                            const Scal a ( (four * ubarBu) * u[k][i] - two * v[i]);
                            const Scal b ( two * u[k][i] );
                            
                            for( Int j = i; j < n; ++j )
                            {
                                B[i][j] += a * Conj(u[k][j]) - b * Conj(v[j]);
                            }
                        }
                    }
                                       
                    // We want a purely real tridiagonal matrix...
                    for( Int i = 0; i < n-1; ++i )
                    {
                        T.Diag(i)  = Re(B[i][i]);
                        T.Upper(i) = ( Scalar::ComplexQ<Scal> ? Abs(B[i][i+1]) : B[i][i+1] );
                    }
                    T.Diag(n-1)  = Re(B[n-1][n-1]);
                }
            }
            
            void Eigenvalues( 
                Vector<n,Real,Int> & eigs,
                const Real tol      = eps,
                const Int  max_iter = 16
            ) const
            {
                SelfAdjointTridiagonalMatrix<n, Real, Int> T;
                
                HessenbergDecomposition(T);
                
                T.QRAlgorithm(eigs,tol,max_iter);
            }
            
            
            std::pair<Matrix<n,n,Scal,Int>,Vector<n,Real,Int>> Eigensystem(
                const Real tol      = eps,
                const Int  max_iter = 16
            ) const
            {
                // Returns U and eigs such that
                // ConjugateTranspose(U) * A * U == DiagonalMatrix(eigs);
                // That means, the COLUMNS of U are the eigenvectors.
                
                std::pair<Matrix<n,n,Scal,Int>,Vector<n,Real,Int>> result;
                
                Eigensystem(result.first,result.second,tol,max_iter);
                
                return result;
            }
            
            void Eigensystem(
                Matrix<n,n,Scal,Int> & U,
                Vector<n,  Real,Int> & eigs,
                const Real tol      = eps,
                const Int  max_iter = 16
            ) const
            {
                // Returns U and eigs such that
                // ConjugateTranspose(U) * A * U == DiagonalMatrix(eigs);
                // That means, the COLUMNS of U are the eigenvectors.
                
                SelfAdjointTridiagonalMatrix<n, Real, Int> T;
                
                Matrix<n,n,Scal,Int> V;
                Matrix<n,n,Real,Int> Q;
                
                HessenbergDecomposition(V,T);

                T.QRAlgorithm(Q,eigs,tol,max_iter);

                //TODO: We might exploit here that V has zeroes in first row and column.
                Dot<Overwrite>(V,Q,U);
            }
            
            TOOLS_FORCE_INLINE Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < n; ++i )
                {
                    AA += AbsSquared(A[i][i]);
                    
                    for( Int j = i+1; j < n; ++j )
                    {
                        AA += 2 * AbsSquared(A[i][j]);
                    }
                }
                return Sqrt(AA);
            }
            
            template<typename T = Scal>
            void ToMatrix( Matrix<n,n,T,Int> & B ) const
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < i; ++j )
                    {
                        B[i][j] = static_cast<T>(Conj(A[j][i]));
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        B[i][j] = static_cast<T>(A[i][j]);
                    }
                }
            }
            
            Matrix<n,n,Scal,Int> ToMatrix() const
            {
                Matrix<n,n,Scal,Int> B;
                
                ToMatrix(B);
                
                return B;
            }
            
            static std::string ClassName()
            {
                return ct_string("Tiny::SelfAdjointMatrix")
                    + "<" + to_ct_string(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + ">";
            }
            
        };
        
        
        template<int m_, int n_, typename Scal, IntQ Int >
        [[nodiscard]] TOOLS_FORCE_INLINE const
        SelfAdjointMatrix<n_,Scal,Int> SelfAdjointAHA( const Matrix<m_,n_,Scal,Int> & A )
        {
            return A.template AHA<true>();
        }
        
        template<int m_, int n_, typename Scal, IntQ Int >
        [[nodiscard]] TOOLS_FORCE_INLINE const
        SelfAdjointMatrix<m_,Scal,Int> SelfAdjointAAH( const Matrix<m_,n_,Scal,Int> & A )
        {
            return A.template AAH<true>();
        }
        
    } // namespace Tiny
    
} // namespace Tensors
