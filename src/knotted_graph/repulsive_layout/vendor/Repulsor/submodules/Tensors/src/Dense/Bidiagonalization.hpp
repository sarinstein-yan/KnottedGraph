#pragma once

// Not production ready. Just a few thoughts on the design of a thin wrapper class that would maybe make calling BLAS routines a bit easier.

namespace Tensors
{
    namespace Dense
    {
        
        
        template<typename Scal_, IntQ Int_, UpLo uplo_>
        class BidiagonalMatrix
        {
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            
            static constexpr UpLo uplo = uplo_;
            
            
            explicit BidiagonalMatrix( const Int k_ )
            :   k ( k_ )
            ,   diag { {k}, {k-Int(1)} };
            {}
            
        private:
            
            const Int k = 0;
            
            Tensor1<Scal,Int> diag [2];
            
        public:
            
            Int RowCount() const
            {
                return k;
            }
            
            Int ColCount() const
            {
                return k;
            }
            
            mref<Tensor1<Scal,Int>> Diagonal( const bool i )
            {
                return diag[i];
            }
            
            cref<Tensor1<Scal,Int>> Diagonal( const bool i ) const
            {
                return diag[i];
            }
            
            static constexpr bool UpperQ()
            {
                return (uplo == UpLo::Upper);
            }
            
            static constexpr bool LowerQ()
            {
                return (uplo == UpLo::Lower);
            }
        }
        
        // Computes the vector v whose Householder reflector maps vector a to vector e_k.
        template<typename Scal, IntQ Int>
        void HouseholderVector(
            const Int n, cptr<Scal> x, const Int inc_x, const Int k, mptr<Scal> v
        )
        {
            // cf. https://en.wikipedia.org/wiki/Householder_transformation
            
            // x - 2 * <x,v> * v = a * e_k
            //
            // 2 * <x,v> * v = x - a * e_k
            //
            // v must be a unit vector, hence
            //
            // v = +/  (x - a * e_k) / |x - a * e_k|
            
            const Scal squared_norm_x = 0;
            
            for( Int i = 0; i < n; ++i )
            {
                v[i] = x[inc_x * i];
                squared_norm += v[i] * v[i];
            }
            
            const Scal a = Sign( Re(x[k]) ) * Sqrt(squared_norm_x);
            
            squared_norm_x -= v[k] * v[k];
            v[k] += a;
            squared_norm_x += v[k] * v[k];
            
            scale_buffer( Inv<Scal>(Sqrt(squared_norm_x)), x, n );
        }
        
        
        // Returns factorization A = U * Delta * V^*.
        // Delta is upper bidiagonal if m >= n and lower bidiagonal otherwise.
        // Only the diagonals are returned, not Delta.
        // The input matrix A is overwritten.
        template<typename Scal, IntQ Int>
        std::tuple<Tensor2<Scal,Int>,Tensor1<Scal,Int>,Tensor1<Scal,Int>,Tensor2<Scal,Int>>
        Bidiagonalization( mptr<Scal> A, const Int m, const Int n )
        {
            const Int k = Min(m,n);
            
            Tensor2<Scal,Int> u (k,m); // Householder vectors
            Tensor2<Scal,Int> v (k,n); // Householder vectors
            
            Tensor1<Scal,Int> diag_0 (k       );
            Tensor1<Scal,Int> diag_1 (k-Int(1));
                        
            
            if( m >= n )
            {
//                HouseholderVector( m, A, n, Int(1), u.data(1,0) );
            }
            else // m < n
            {
            }
        }
        
        
    } namespace Dense
    
} // namespace Tensors
