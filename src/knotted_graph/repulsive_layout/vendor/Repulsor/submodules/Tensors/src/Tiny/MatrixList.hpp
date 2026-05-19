#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template< int m_, int n_, typename Scal_, IntQ Int_, Size_T alignment = CacheLineWidth>
        class MatrixList final
        {
        public:
            
#include "Tiny_Constants.hpp"
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            static constexpr Size_T Alignment = alignment;
            
            using Tensor_T = Tensor1<Scal,Int,Alignment>;
            
        private:
            
            Int length = 0;
            
            Tensor_T A [m][n];
            
        public:
            //  The big four and half:
            
            MatrixList() = default;
            
            //Destructor
            ~MatrixList() noexcept = default;
            
            explicit MatrixList( const Int length_ )
            :   length(length_)
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = Tensor_T(length_);
                    }
                }
            }
            
            MatrixList( const Int length_, cref<Scal> init )
            :   length(length_)
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = Tensor_T(length_,init);
                    }
                }
            }
            
            // Copy constructor
            MatrixList( const MatrixList & other )
            :   MatrixList( other.length )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j].Read( other.A[i][j].data());
                    }
                }
            }
            
            friend void swap( MatrixList & X, MatrixList & Y ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                std::swap( X.length, Y.length );
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        swap( X.A[i][j], Y.A[i][j] );
                    }
                }
            }
            
            // Copy assignment operator
            MatrixList & operator=( MatrixList other ) noexcept
            {   //                                ^
                //                                |
                // Use the copy constructor ------+
                swap( *this, other );
                return *this;
            }
            
            // Move constructor
            MatrixList( MatrixList && other ) noexcept
            :   MatrixList()
            {
                swap(*this, other);
            }
            
            
            //  Access routines
            
            mptr<Scal> data( const Int i, const Int j ) noexcept
            {
                return A[i][j].data();
            }
            
            cptr<Scal> data( const Int i, const Int j ) const noexcept
            {
                return A[i][j].data();
            }
            
            constexpr void SetZero()
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j].SetZero();
                    }
                }
            }
            
            mref<Tensor_T> operator()( const Int i, const Int j ) noexcept
            {
                return A[i][j];
            }
            
            const Tensor_T & operator()( const Int i, const Int j ) const noexcept
            {
                return A[i][j];
            }
            
            mref<Scal> operator()( const Int i, const Int j, const Int k ) noexcept
            {
                return A[i][j][k];
            }
            
            cref<Scal> operator()( const Int i, const Int j, const Int k ) const noexcept
            {
                return A[i][j][k];
            }
            
//            mref<std::array<Tensor_T,n>> operator[]( const Int i ) noexcept
//            {
//                return A[i];
//            }
//            
//            cref<std::array<Tensor_T,n>> operator[]( const Int i ) const noexcept
//            {
//                return A[i];
//            }
            
            mptr<Tensor_T> operator[]( const Int i ) noexcept
            {
                return &A[i];
            }
            
            cptr<Tensor_T> operator[]( const Int i ) const noexcept
            {
                return &A[i];
            }
            
            
            template<typename S>
            void Read( const S * const * const * const a )
            {
                //Assuming that a is a list of m x n pointers pointing to memory of at least size Dim(1).
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        copy_buffer( a[i][j], &A[i][j], length );
                    }
                }
            }
            
            template<typename S>
            void Write( S * const * const * const a ) const
            {
                //Assuming that a is a list of m pointers pointing to memory of at least size Dim(1).
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        copy_buffer( &A[i][j], a[i][j], length );
                    }
                    
                }
            }
            
            template<typename S>
            void Read( const S * const a_ )
            {
                //Assuming that a is a list of size Dim(1) x m of vectors in interleaved form.
                
                for( Int k = 0; k < length; ++ k)
                {
                    for( Int i = 0; i < m; ++ i)
                    {
                        for( Int j = 0; j < n; ++j )
                        {
                            A[i][j][k] = a_[(k*m+i)*n+j];
                        }
                    }
                }
            }
            
            template<typename S>
            void Write( S * const a ) const
            {
                //Assuming that a is a list of size Dim(1) x m of vectors in interleaved form.
                
                for( Int k = 0; k < length; ++k )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        for( Int j = 0; j < n; ++j )
                        {
                            a[(k*m+i)*n+j] = A[i][j][k];
                        }
                    }
                }
            }
            
        public:
            
            static constexpr Int Rank() noexcept
            {
                return 3;
            }
            
            Int Dim( const Int k ) const noexcept
            {
                switch( k )
                {
                    case 0:
                    {
                        return m;
                    }
                    case 1:
                    {
                        return n;
                    }
                    case 2:
                    {
                        return A[0][0].Dim(0);
                    }
                    default:
                    {
                        return 0;
                    }
                }
            }
            
            Int Dimension( const Int k ) const noexcept
            {
                return Dim(k);
            }
            
            Size_T AllocatedByteCount() const
            {
                Size_T b = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        b += A[i][j].AllocatedByteCount();
                    }
                }
                
                return b;
            }
            
            Size_T ByteCount() const
            {
                return sizeof(MatrixList) + AllocatedByteCount();
            }
            
            
            auto WriteAccess()
            {
                return [this]( const Int i, const Int j, const Int k ) -> Scal&
                {
                    return A(i,j,k);
                };
            }
            
            auto ReadAccess() const
            {
                return [this]( const Int i, const Int j, const Int k ) -> Scal
                {
                    return A(i,j,k);
                };
            }
            
            inline friend std::ostream & operator<<( std::ostream & s, cref<MatrixList> tensor )
            {
                return s << OutString::FromCube(tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1), tensor.Dim(2));
            }
            
            inline friend std::string ToString( cref<MatrixList> tensor )
            {
                return OutString::FromCube(tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1), tensor.Dim(2));
            }
            
            inline friend std::string ToString( cref<MatrixList> tensor, cref<std::string> line_prefix )
            {
                return OutString::FromArray(
                    tensor.ReadAccess(),
                    tensor.Dim(0), line_prefix + "{\n", ",\n", "\n" + line_prefix + "}",
                    tensor.Dim(1), line_prefix + " { ", ", ", " }",
                    tensor.Dim(2),               "{ ", ", ", " }"
                );
            }
            
            
            [[nodiscard]] static std::string ClassName() noexcept
            {
                return ct_string("Tiny::MatrixList")
                + "<" + to_ct_string(m)
                + "," + to_ct_string(n)
                + "," + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + to_ct_string(Alignment)
                + ">";
            }
        };
        
        
    #ifdef LTEMPLATE_H
        
        
        template<int m, int n, std::floating_point T, typename I>
        inline mma::TensorRef<mreal> to_MTensorRef( cref<Tiny::MatrixList<m,n,T,I>> A )
        {
            const mint N = A.Dim(2);
            
            const T * restrict p [m][n];
            
            for( mint i = 0; i < m; ++i )
            {
                for( mint j = 0; j < n; ++j )
                {
                    p[i][j] = A.data(i,j);
                }
            }
            
            auto B = mma::makeCube<mreal>( N, m, n );
            
            mptr<mreal> b = B.data();
            
            for( mint k = 0; k < N; ++k )
            {
                for( mint i = 0; i < m; ++i )
                {
                    for( mint j = 0; j < n; ++j )
                    {
                        b[(m * k + i) * n + j] = static_cast<mreal>(p[i][j][k]);
                    }
                }
            }
            
            return B;
        }
        
        template<int m, int n, IntQ J, IntQ I>
        inline mma::TensorRef<mint> to_MTensorRef( cref<Tiny::MatrixList<m,n,J,I>> A )
        {
            const mint N = A.Dim(2);
            
            const J * restrict p [m][n];
            
            for( mint i = 0; i < m; ++i )
            {
                for( mint j = 0; j < n; ++j )
                {
                    p[i][j] = A.data(i,j);
                }
            }
            
            auto B = mma::makeCube<mint>( N, m, n );
            
            mptr<mint> b = B.data();
            
            for( mint k = 0; k < N; ++k )
            {
                for( mint i = 0; i < m; ++i )
                {
                    for( mint j = 0; j < n; ++j )
                    {
                        b[(m * k + i) * n + j] = static_cast<mreal>(p[i][j][k]);
                    }
                }
            }
            
            return B;
        }
        
    #endif
    
    } // namespace Tiny
    
} // namespace Tensors
