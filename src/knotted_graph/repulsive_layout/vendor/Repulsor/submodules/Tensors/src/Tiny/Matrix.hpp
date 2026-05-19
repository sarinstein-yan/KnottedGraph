#pragma once

#include <initializer_list>

namespace Tensors
{
    namespace Tiny
    {
        template< int ROW_COUNT, typename Scal_, IntQ Int_> class SelfAdjointMatrix;
        
        template< int ROW_COUNT, int COL_COUNT, typename Scal_, IntQ Int_, Size_T alignment> class MatrixList;
        
        template< int ROW_COUNT, int COL_COUNT, typename Scal_, IntQ Int_>
        class Matrix final
        {
        public:
            
            using Class_T = Matrix;
            
#include "Tiny_Constants.hpp"
            
            static constexpr Int m = ROW_COUNT;
            static constexpr Int n = COL_COUNT;
            
            using ColVector_T = Vector<m,Scal,Int>;
            using RowVector_T = Vector<n,Scal,Int>;
            
            using Vector_T    = Vector<n,Scal,Int>;
            
        protected:
            
            Scal A[Tools::Max(Int(1),m)][Tools::Max(Int(1),n)];
            
        public:
            
            Matrix(std::nullptr_t) = delete;

            explicit Matrix( const Scal * a )
            {
                Read(a);
            }
            
            explicit Matrix( cref<Scal> init )
            :   A {{{init}}}
            {}
            
            template<typename S>
            constexpr Matrix( const std::initializer_list<S[n]> list )
            {
                const Int m_ = static_cast<Int>(list.size());
                
                auto iter { list.begin() };
                for( Int i = 0; i < m_; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = static_cast<Scal>((*iter)[j]);
                    }
                    ++iter;
                }
                
                for( Int i = m_; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = 0;
                    }
                }
            }
            
            // Default constructor
            Matrix() = default;
            
            // Destructor
            ~Matrix() = default;
            
            // Copy constructor
            Matrix( const Matrix & other ) noexcept
            {
                Read( &other.A[0][0] );
            }

            // Copy assignment operator
            Matrix & operator=( const Matrix & other ) noexcept
            {
                Read( &other.A[0][0] );

                return *this;
            }
            
            // Move constructor
            Matrix( Matrix && other ) noexcept
            {
//                swap(*this, other);
                Read( &other.A[0][0] );
            }
            
            // Move assignment operator
            Matrix & operator=( Matrix && other ) noexcept
            {
                Read( &other.A[0][0] );
                return *this;
            }

            
            template<typename S, Size_T alignment>
            Matrix( cref<MatrixList<m,n,S,Int,alignment>> m_list, const Int k )
            {
                Read(m_list, k);
            }
            
            template<typename S>
            Matrix( cref<Tensor3<S,Int>> tensor, const Int k )
            {
                Read(tensor.data(k));
            }

            
//######################################################
//##                     Access                       ##
//######################################################

#include "Tiny_Matrix_Common.hpp"
            
///######################################################
///##                     Memory                       ##
///######################################################
                
        public:

            void SetZero()
            {
                zerofy_buffer<m*n>( &A[0][0] );
            }
            
            template<typename T>
            void Fill( cref<T> init )
            {
                fill_buffer<m*n>( &A[0][0], static_cast<T>(init) );
            }


            template<typename B_T>
            void AddTo( mptr<B_T> B ) const
            {
                add_to_buffer<m*n>( &A[0][0], B );
            }
                
            
            void WriteRow( mref<RowVector_T> u, const Int i )
            {
                u.Read( &A[i] );
            }
            
            void WriteCol( mref<ColVector_T> v, const Int j )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][j];
                }
            }
            
            
///######################################################
///##             Reading from raw buffers             ##
///######################################################

            /// BLAS-like read-modify method _with stride_.
            /// Reads the full matrix.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Read(
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB
            )
            {
                /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
                /// `opA` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opB) )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        combine_buffers<beta_flag,alpha_flag,n,Sequential,op,opA>(
                            beta, &B[ldB*i], alpha, &A[i][0]
                        );
                    }

                }
                else if constexpr ( TransposedQ(opB) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        cptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<beta_flag,alpha_flag,op,opA>(
                                beta, B_j[i], alpha, A[i][j]
                            );
                        }
                    }
                }
            }

            /// BLAS-like read-modify method _without stride_.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Read(
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B
            )
            {
                /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
                /// `opA` can only be `Op::Id` or `Op::Conj`.

                constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opB) )
                {
                    combine_buffers<beta_flag,alpha_flag,m*n,Sequential,op,opA>(
                        beta, B, alpha, &A[0][0]
                    );
                }
                else if constexpr ( TransposedQ(opB) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        cptr<B_T> B_j = &B[m * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<beta_flag,alpha_flag,op,opA>(
                                beta, B_j[i], alpha, A[i][j]
                            );
                        }
                    }
                }
            }

            /// BLAS-like read-modify method _without stride_.
            template<Op opB = Op::Id, typename B_T>
            void Read( cptr<B_T> B )
            {
                /// Compute `opB(B)` and store it in `A`.
                
                Read<Scalar::Flag::Zero,Scalar::Flag::Plus,Op::Id,opB>(
                    Scal(0), Scal(1), B
                );
            }

            /// BLAS-like read-modify method _with stride_.
            template<Op opB = Op::Id, typename B_T>
            void Read( cptr<B_T> B, const Int ldB )
            {
                /// Compute `opB(B)` and store it in `A`.
                
                Read<Scalar::Flag::Zero,Scalar::Flag::Plus,Op::Id,opB>(
                    Scal(0), Scal(1), B, ldB
                );
            }


            /// BLAS-like read-modify method _with stride_.
            /// Reads only the top left portion.
            /// It is meant to read from the right and bottom boundaries of a large matrix.
            template<
                bool chop_m_Q, bool chop_n_Q,
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void ReadChopped(
                const Int m_max, const Int n_max,
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB = n
            )
            {
                /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
                /// `opA` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
                
                const Int m_c = Min(m,m_max);
                const Int n_c = Min(n,n_max);

                if constexpr ( NotTransposedQ(opB) )
                {
                    for( Int i = 0; i < (chop_m_Q ? m_c : m); ++i )
                    {
                        combine_buffers<beta_flag,alpha_flag,(chop_n_Q ? 0 : n),Sequential,op,opA>(
                            beta, &B[ldB*i], alpha, &A[i][0], n_c
                        );
                    }
                }
                else if constexpr ( TransposedQ(opB) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    
                    /// TODO:
                    /// I think best performance should be obtained
                    /// by making the fixed-size loop the inner loop.
                    
                    for( Int j = 0; j < (chop_n_Q ? n_c : n); ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < (chop_m_Q ? m_c : m); ++i )
                        {
                            combine_scalars<beta_flag,alpha_flag,op,opA>(
                                beta, B_j[i], alpha, A[i][j]
                            );
                        }
                    }
                }
            }

        //    // Scattered read-modify method.
        //    // Might be useful in supernodal arithmetic for sparse matrices.
        //    template<Scalar::Flag a_flag, Op op, typename alpha_T, typename B_T>
        //    void Read( const alpha_T alpha, cptr<B_T> B, const Int ldB, cptr<Int> idx  )
        //    {
        //        // Reading A = alpha * op(B)
        //
        //        if constexpr ( op == Op::Id )
        //        {
        //            if constexpr ( a_flag == Scalar::Flag::Plus )
        //            {
        //                for( Int i = 0; i < m; ++i )
        //                {
        //                    copy_buffer<n>( &B[ldB*idx[i]], &A[i][0] );
        //                }
        //            }
        //        }
        //        else if constexpr ( op == Op::Conj )
        //        {
        //            for( Int i = 0; i < m; ++i )
        //            {
        //                for( Int j = 0; j < n; ++j )
        //                {
        //                    A[i][j] = ScalarOperator<a_flag,op>( alpha, B[i][j] );
        //                }
        //            }
        //        }
        //        else
        //        {
        //            // TODO: Not sure whether it would be better to swap the two loops here...
        //            for( Int j = 0; j < n; ++j )
        //            {
        //                cptr<B_T> B_j = &B[ldB*idx[j]];
        //
        //                for( Int i = 0; i < m; ++i )
        //                {
        //                    A[i][j] = ScalarOperator<a_flag,op>( alpha, B_j[i] );
        //                }
        //            }
        //        }
        //    }

        ///######################################################
        ///##              Writing to raw buffers              ##
        ///######################################################

            /// BLAS-like write-modify method _with stride_.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Write(
                cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B, const Int ldB
            ) const
            {
                /// Computes `B = alpha * opA(A) + beta * opB(B)`.
                /// `opB` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opA) )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        combine_buffers<alpha_flag,beta_flag,n,Sequential,op,opB>(
                            alpha, &A[i][0], beta, &B[ldB*i]
                        );
                    }
                }
                else if constexpr ( TransposedQ(opA) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    for( Int j = 0; j < n; ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opB>(
                                alpha, A[i][j], beta, B_j[i]
                            );
                        }
                    }
                }
            }



            /// BLAS-like write-modify method _without stride_.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Write(
                cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B
            ) const
            {
                /// Computes `B = alpha * opA(A) + beta * opB(B)`.
                /// `opB` can only be `Op::Id` or `Op::Conj`.

                constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opA) )
                {
                    /// This is the only reason we use a version without stride:
                    /// If the whole matrix is stored contiguously, we can vectorize over
                    /// row-ends!
                    combine_buffers<alpha_flag,beta_flag,m*n,Sequential,op,opB>(
                        alpha, &A[0][0], beta, B
                    );
                }
                else if constexpr ( TransposedQ(opA) )
                {
                    /// I think that no real optimizations can be made here.
                    /// Nonetheless, we let the compiler know that the stride is a
                    /// compile-time constant. But since the alignment of B is unknown to
                    /// the compiler, this might be quite useless.

                    // TODO: Compare with reverse ordering of loops.
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        mptr<B_T> B_j = &B[m * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opB>(
                                alpha, A[i][j], beta, B_j[i]
                            );
                        }
                    }
                }
            }


            /// BLAS-like write-modify method _with stride_.
            template<Op opA = Op::Id, typename B_T>
            void Write( mptr<B_T> B, const Int ldB ) const
            {
                /// Compute `B = opA(A)`.

                Write<Scalar::Flag::Plus,Scalar::Flag::Zero,opA,Op::Id>( 
                    Scalar::One<B_T>, Scalar::Zero<B_T>, B, ldB
                );
            }

            /// BLAS-like write-modify method _without stride_.
            template<Op op, typename B_T>
            void Write( mptr<B_T> B ) const
            {
                /// B = opA(A)
                Write<Scalar::Flag::Plus,Scalar::Flag::Zero,op,Op::Id>(
                    Scalar::One<B_T>, Scalar::Zero<B_T>, B
                );
            }
            
            /// Simple copy routine. (Need, e.g. for integer types.)
            template<typename B_T>
            void Write( mptr<B_T> B ) const
            {
                /// B = A
                copy_buffer<m * n>( &A[0][0], B );
            }

        //    /// Row-scattered write-modify method.
        //    /// Might be useful in supernodal arithmetic for sparse matrices.
        //    template<
        //        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        //        typename alpha_T, typename beta_T, typename B_T
        //    >
        //    void Write( cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B, const Int ldB, cptr<Int> idx ) const
        //    {
        //
        //        // Writing B[idx[i]][j] = alpha * A[i][j] + beta * B[idx[i]][j]
        //        for( Int i = 0; i < m; ++i )
        //        {
        //            combine_buffers<alpha_flag,beta_flag,n>(
        //                alpha, &A[i][0], beta, &B[ldB*idx[i]]
        //            );
        //        }
        //    }

            /// BLAS-like write-modify method _with stride_.
            /// Writes only the top left portion.
            /// It is meant to write to the right and bottom boundaries of a large matrix.
            template<
                bool chop_m_Q, bool chop_n_Q,
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void WriteChopped(
                const Int m_max, const Int n_max,
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB = n
            ) const
            {
                /// Computes `B = alpha * opA(A) + beta * opB(B)`.
                /// `opB` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
                
                const Int m_c = Min( m, m_max );
                const Int n_c = Min( n, n_max );
                
                if constexpr ( NotTransposedQ(opA) )
                {
                    for( Int i = 0; i < (chop_m_Q ? m_c : m); ++i )
                    {
                        combine_buffers<alpha_flag,beta_flag,(chop_n_Q ? 0 : n),Sequential,op,opB>(
                            alpha, &A[i][0], beta, &B[ldB*i], n_c
                        );
                    }
                }
                else if constexpr ( TransposedQ(opA) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    for( Int j = 0; j < (chop_n_Q ? n_c : n); ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < (chop_m_Q ? m_c : m); ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opB>(
                                alpha, A[i][j], beta, B_j[i]
                            );
                        }
                    }
                }
            }
            
///######################################################
///##             Arithmetic with scalars              ##
///######################################################
         
        public:
            
            template<class T>
            TOOLS_FORCE_INLINE mref<Matrix> operator+=( const T lambda_ )
            {
                const auto lambda = scalar_cast<Scal>(lambda_);
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += lambda;
                    }
                }
                
                return *this;
            }

            template<class T>
            TOOLS_FORCE_INLINE mref<Matrix> operator-=( const T lambda_ )
            {
                const auto lambda = scalar_cast<Scal>(lambda_);
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= lambda;
                    }
                }
                
                return *this;
            }
            
            template<class T>
            TOOLS_FORCE_INLINE mref<Matrix> operator*=( const T lambda_ )
            {
                const auto lambda = scalar_cast<Scal>(lambda_);
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= lambda;
                    }
                }
                
                return *this;
            }

            template<class T>
            TOOLS_FORCE_INLINE mref<Matrix> operator/=( const T lambda )
            {
                return (*this) *= ( scalar_cast<Scal>(Inv<T>(lambda)) );
            }
            
///######################################################
///##                   Arithmetic                     ##
///######################################################

            TOOLS_FORCE_INLINE friend bool operator==(
                cref<Matrix> x, cref<Matrix> y
            )
            {
                return buffers_equalQ<m * n>(x.data(), y.data());
            }
            
            // TODO: Make this more type flexible.
            // TODO: Also, where is Minus?
            TOOLS_FORCE_INLINE friend void Plus( const Matrix & x, const Matrix & y, const Matrix & z )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
            }
            
            // TODO: Make this more type flexible.
            // TODO: Also, where is operator-?
            [[nodiscard]] TOOLS_FORCE_INLINE friend const Matrix operator+( const Matrix & x, const Matrix & y )
            {
                Matrix z;
                
                Plus( x, y, z);
                
                return z;
            }
            
            template<class T>
            TOOLS_FORCE_INLINE
            mref<Matrix> operator+=( cref<Tiny::Matrix<m,n,T,Int>> B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            template<class T>
            TOOLS_FORCE_INLINE
            mref<Matrix> operator-=( cref<Tiny::Matrix<m,n,T,Int>> B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
//            // TODO: Not sure whether I want * to represent compontenwise multiplication.
//            template<class T>
//            TOOLS_FORCE_INLINE
//            mref<Matrix> operator*=( cref<Tiny::Matrix<m,n,T,Int>> B )
//            {
//                for( Int i = 0; i < m; ++i )
//                {
//                    for( Int j = 0; j < n; ++j )
//                    {
//                        A[i][j] *= B.A[i][j];
//                    }
//                }
//                
//                return *this;
//            }
//            
//            // TODO: Not sure whether I want / to represent compontenwise division.
//            template<class T>
//            TOOLS_FORCE_INLINE
//            mref<Matrix> operator/=( cref<Tiny::Matrix<m,n,T,Int>> B )
//            {
//                for( Int i = 0; i < m; ++i )
//                {
//                    for( Int j = 0; j < n; ++j )
//                    {
//                        A[i][j] /= B.A[i][j];
//                    }
//                }
//                
//                return *this;
//            }
            
        public:
            
            template<Op op>
            void LowerFromUpper()
            {
                for( Int i = 0; i < n; ++i )
                {
                    // TODO: Unroll this loop trough templates?
                    for( Int j = 0; j < i; ++j )
                    {
                        A[i][j] = ScalarOperator<Scalar::Flag::Plus,op>(A[j][i]);
                    }
                }
            }
           
        public:

            template<bool upper_triangle_only = false>
            [[nodiscard]] TOOLS_FORCE_INLINE std::conditional_t<
                upper_triangle_only,
                SelfAdjointMatrix<n,Scal,Int>,
                Matrix<n,n,Scal,Int>
            > AHA() const
            {
                std::conditional_t<
                    upper_triangle_only,
                    SelfAdjointMatrix<n,Scal,Int>,
                    Matrix<n,n,Scal,Int>
                > B;
                
                
                // TODO: Not sure whether this is faster than using Dot + ConjugateTranspose.
                {
                    constexpr Int k = 0;
                    
                    for( Int i = 0; i < n; ++i )
                    {
                        const Scal Conj_A_ki = Conj(A[k][i]);
                        
                        // TODO: Unroll this loop trough templates?
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] = Conj_A_ki * A[k][j];
                        }
                    }
                    
                }
                
                for( Int k = 1; k < m; ++k )
                {
                    for( Int i = 0; i < n; ++i )
                    {
                        const Scal Conj_A_ki = Conj(A[k][i]);
                        
                        // TODO: Unroll this loop trough templates?
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] += Conj_A_ki * A[k][j];
                        }
                    }
                }

                B.template LowerFromUpper<Op::Conj>();
                
                return B;
            }
            
            template<bool upper_triangle_only = false>
            [[nodiscard]] TOOLS_FORCE_INLINE std::conditional_t<
                upper_triangle_only,
                SelfAdjointMatrix<m,Scal,Int>,
                Matrix<m,m,Scal,Int>
            > AAH() const
            {
                std::conditional_t<
                    upper_triangle_only,
                    SelfAdjointMatrix<m,Scal,Int>,
                    Matrix<m,m,Scal,Int>
                > B;
                
                for( Int i = 0; i < m; ++i )
                {
                    B[i][i] = dot_buffers<n,Sequential,Op::Id,Op::Conj>( &A[i][0], &A[i][0] );
                    
                    // TODO: Unroll this loop trough templates?
                    for( Int j = i + 1; j < m; ++j )
                    {
                        B[i][j] = dot_buffers<n,Sequential,Op::Id,Op::Conj>( &A[i][0], &A[j][0] );
                    }
                }
                
                B.template LowerFromUpper<Op::Conj>();
                
                return B;
            }

        public:
            
            
            TOOLS_FORCE_INLINE void Conjugate( Matrix & B ) const
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        B.A[i][j] = Conj(A[i][j]);
                    }
                }
            }

            [[nodiscard]] TOOLS_FORCE_INLINE Matrix Conjugate() const
            {
                Matrix B;
                
                Conjugate(B);
                
                return B;
            }
            
        public:
            
            TOOLS_FORCE_INLINE void Transpose( mref<Matrix<n,m,Scal,Int>> B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B[j][i] = A[i][j];
                    }
                }
                
                // TODO: Test whether the following is as fast; if yes, then simply.
//                Write<Op::Trans,Op::Id>( &B[0][0] );
            }
            
            [[nodiscard]] TOOLS_FORCE_INLINE Matrix<n,m,Scal,Int> Transpose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                Transpose(B);
                
                return B;
            }

            TOOLS_FORCE_INLINE void ConjugateTranspose( mref<Matrix<n,m,Scal,Int>> B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B[j][i] = Conj(A[i][j]);
                    }
                }
                
                // TODO: Test whether the following is as fast; if yes, then simply.
                
//                Write<Op::ConjTrans,Op::Id>( &B[0][0] );
            }

            [[nodiscard]] TOOLS_FORCE_INLINE Matrix<n,m,Scal,Int> ConjugateTranspose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                ConjugateTranspose(B);
                
                return B;
            }
            
            
            
            [[nodiscard]] TOOLS_FORCE_INLINE Real MaxNorm() const
            {
                return norm_max<m*n>( &A[0][0] );
            }
            
            [[nodiscard]] TOOLS_FORCE_INLINE Real AbsTotal() const
            {
                return norm_1<m*n>( &A[0][0] );
            }
            
            [[nodiscard]] TOOLS_FORCE_INLINE Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        AA += AbsSquared(A[i][j]);
                    }
                }
                
                return Sqrt(AA);

                // TODO: Test whether the following is as fast; if yes, then simply.
//                norm_2<m*n>(&A[0][0]);
            }
            
        public:
            
            void Threshold( const Real threshold )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        if( Abs(A[i][j]) <= threshold )
                        {
                            A[i][j] = static_cast<Scal>(0);
                        }
                    }
                }
            }
            
        public:
            
            template<class T>
            TOOLS_FORCE_INLINE void GivensLeft( const T c_, const T s_, const Int i, const Int j )
            {
                if constexpr ( n >= Int(2) )
                {
                    const Scal c = scalar_cast<Scal>(c_);
                    const Scal s = scalar_cast<Scal>(s_);
                    
                    /* Assumes that AbsSquared(c) + AbsSquared(s) == one.
                     * Assumes that i != j.
                     * Multiplies matrix with the rotation
                     *
                     *    /               \
                     *    |      c     s  |
                     *    |  -conj(s)  c  |
                     *    \               /
                     *
                     * in the i-j-plane from the left.
                     */
                    
                    for( Int k = 0; k < n; ++k )
                    {
                        const Scal x = A[i][k];
                        const Scal y = A[j][k];
                        
                        A[i][k] =      c    * x + s * y;
                        A[j][k] = - Conj(s) * x + c * y;
                    }
                }
            }

            template<class T>
            TOOLS_FORCE_INLINE void GivensRight( const T c_, const T s_, const Int i, const Int j )
            {
                if constexpr ( n >= Int(2) )
                {
                    const Scal c = scalar_cast<Scal>(c_);
                    const Scal s = scalar_cast<Scal>(s_);
                    
                    /*
                     * Assumes that AbsSquared(c) + AbsSquared(s) == one.
                     * Assumes that i != j.
                     * Multiplies matrix with rotation
                     *
                     *    /               \
                     *    |      c     s  |
                     *    |  -conj(s)  c  |
                     *    \               /
                     *
                     * in the i-j-plane from the right.
                     */
                    
                    for( Int k = 0; k < m; ++k )
                    {
                        const Scal x = A[k][i];
                        const Scal y = A[k][j];
                        
                        A[k][i] = c * x - Conj(s) * y;
                        A[k][j] = s * x +    c    * y;
                    }
                }
            }
            
            
            
        public:
           
            constexpr TOOLS_FORCE_INLINE void SetIdentity()
            {
                static_assert(m==n, "SetIdentity is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = KroneckerDelta<Real>(i,j);
                    }
                }
            }
            
            TOOLS_FORCE_INLINE void MakeDiagonal( const Tensors::Tiny::Vector<n,Scal,Int> & v )
            {
                static_assert(m==n, "MakeDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? v[i] : static_cast<Scal>(0);
                    }
                }
            }
            
            TOOLS_FORCE_INLINE void SetDiagonal( const Tensors::Tiny::Vector<n,Scal,Int> & v )
            {
                static_assert(m==n, "SetDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    A[i][i] = v[i];
                }
            }
            
            [[nodiscard]] Scal Det() const
            {
//                Scal M [n][n];
//
//                Write( &M[0][0] );
//                
//                return Det_Bareiss(n, &M[0][0], n);
                
                if constexpr ( m != n )
                {
                    return 0;
                }
                
                if constexpr ( n == 1 )
                {
                    return A[0][0];
                }
                
                if constexpr ( n == 2 )
                {
                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
                }
                
                if constexpr ( n == 3 )
                {
                    return (
                          A[0][0] * A[1][1] * A[2][2]
                        + A[0][1] * A[1][2] * A[2][0]
                        + A[0][2] * A[1][0] * A[2][1]
                        - A[0][0] * A[1][2] * A[2][1]
                        - A[0][1] * A[1][0] * A[2][2]
                        - A[0][2] * A[1][1] * A[2][0]
                    );
                }
                
                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
                
                if constexpr ( n > Int(3) )
                {
                    Scal M [n][n];
                    
                    Write( &M[0][0] );
                    
                    Scal sign (one);
                    
                    for(Int k = 0; k < n - Int(1); ++k )
                    {
                        //Pivot column swap.
                        
                        const Int l = k + iamax_buffer( &M[k][k], n-k );
                        
                        if( l != k )
                        {
                            sign = -sign;
                            
                            for( Int i = 0; i < n; ++i )
                            {
                                std::swap(M[i][k],M[i][l]);
                            }
                        }
                        
                        const Scal A_k_k = M[k][k];
                        
                        if( A_k_k == zero )
                        {
                            return zero;
                        };
                            
                        
                        // When not working with integers, we want to compute 1/a only once.
                        
                        
                        const Scal a = std::numeric_limits<Scal>::is_integer ? M[k-1][k-1] : Inv(M[k-1][k-1]);
                        
                        
                        //Apply formula
                        for( Int i = k + 1; i < n; ++i )
                        {
                            for( Int j = k + 1; j < n; ++j )
                            {
                                M[i][j] = M[k][k] * M[i][j] - M[i][k] * M[k][j];
                                if(k != 0)
                                {
                                    if constexpr( std::numeric_limits<Scal>::is_integer )
                                    {
                                        M[i][j] /= a;
                                    }
                                    else
                                    {
                                        M[i][j] *= a;
                                    }
                                }
                            }
                        }
                    }
                    
                    return sign * M[n-1][n-1];
                }
                
                return static_cast<Scal>(0);
            }
            
            void LUDecomposition_PivotFree()
            {
                for( Int k = 0; k < std::min(m,n)-1; ++k )
                {
                    const Scal A_kk_inv = Inv( A[k][k] );
                    
                    for( Int i = k+1; i < m; ++i )
                    {
                        A[i][k] *= A_kk_inv;
                        
                        for( Int j = k+1; j < n; ++j )
                        {
                            A[i][j] -= A[i][k] * A[k][j];
                        }
                    }
                }
            }
            
            
        public:
            
            TOOLS_FORCE_INLINE void SetHouseholderReflector(
                const Vector_T & u,
                const Int begin,
                const Int end
            )
            {
                static_assert(m==n, "SetHouseholderReflector is only defined for square matrices.");
                
                // Write the Householder reflection of u into the matrix; assumes that u is zero outside [begin,...,end[.
                
                // Mostly meant for debugging purposes, thus not extremely optimized.
                
                SetIdentity();

                for( Int i = begin; i < end; ++i )
                {
                    for( Int j = begin; j < end; ++j )
                    {
                        A[i][j] -= two * u[i] * Conj(u[j]);
                    }
                }
            }
            
            
            TOOLS_FORCE_INLINE void SetGivensRotation( const Scal c, const Scal s, const Int i, const Int j )
            {
                static_assert(m==n, "SetGivensRotation is only defined for square matrices.");
                
                /*
                 * Mostly meant for debugging purposes, thus not extremely optimized.
                 * Assumes that AbsSquared(c) + AbsSquared(s) == one.
                 * Write Givens rotion
                 *
                 *    /              \
                 *    |     c     s  |
                 *    | -conj(s)  c  |
                 *    \              /
                 *
                 * in the i-j-plane into the matrix.
                 */
                
                SetIdentity();
                
                A[i][i] = c;
                A[i][j] = s;
                A[j][i] = -Conj(s);
                A[j][j] = c;
            }
            
            void Diagonal( Vector<n,Scal,Int> & v ) const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][i];
                }
            }
            
            [[nodiscard]] Vector<n,Scal,Int>  Diagonal() const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                Vector<n,Scal,Int> v;
                Diagonal(v);
                return v;
            }
            
            
            [[nodiscard]] friend std::string ToString( cref<Class_T> M )
            {
                return OutString::FromMatrix(
                    [&M]( const Int i, const Int j ) { return M.A[i][j]; }, m, n
                );
            }

            inline friend std::ostream & operator<<( std::ostream & s, cref<Class_T> M )
            {
                return s << OutString::FromMatrix(
                    [&M]( const Int i, const Int j ) { return M.A[i][j]; }, m, n
                );
            }
            
        public:

            static constexpr Int RowCount()
            {
                return m;
            }
            
            static constexpr Int ColCount()
            {
                return n;
            }
            
            static constexpr Int Dim( const Int i )
            {
                if( i == 0 )
                {
                    return m;
                }
                if( i == 1 )
                {
                    return n;
                }
                return Int(0);
            }
            
            static constexpr Int Dimension( const Int i )
            {
                return Dim(i);
            }
            
            [[nodiscard]] static std::string ClassName()
            {
                return ct_string("Tiny::Matrix")
                    + "<" + to_ct_string(m)
                    + "," + to_ct_string(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + ">";
            }
            
        }; // Tiny::Matrix
                
        
        
        template<AddTo_T addto,
            int m, int k, int n, typename X_T, typename Y_T, typename Z_T, IntQ Int
        >
        TOOLS_FORCE_INLINE void
        Dot(
            cref<Tiny::Matrix<m,k,X_T,Int>> X,
            cref<Tiny::Matrix<k,n,Y_T,Int>> Y,
            mref<Tiny::Matrix<m,n,Z_T,Int>> Z
        )
        {
            fixed_dot_mm<m,n,k,addto>( &X[0][0], &Y[0][0], &Z[0][0] );
        }
        
        template<int m, int K, int n, typename X_T, typename Y_T, IntQ Int>
        [[nodiscard]] TOOLS_FORCE_INLINE
        Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int>
        Dot(
            cref<Tiny::Matrix<m,K,X_T,Int>> X,
            cref<Tiny::Matrix<K,n,Y_T,Int>> Y
        )
        {
            Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int> Z;
         
            Dot<Overwrite>(X,Y,Z);
            
            return Z;
        }
        
        template<int m, int K, int n, typename X_T, typename Y_T, IntQ Int>
        [[nodiscard]] TOOLS_FORCE_INLINE
        Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int>
        operator*(
            cref<Tiny::Matrix<m,K,X_T,Int>> X,
            cref<Tiny::Matrix<K,n,Y_T,Int>> Y
        )
        {
            return Dot(X,Y);
        }
        
        template<AddTo_T addto, int m, int n, typename A_T, typename x_T, typename y_T, IntQ Int
        >
        TOOLS_FORCE_INLINE void
        Dot(
            cref<Tiny::Matrix<m,n,A_T,Int>> A,
            cref<Tiny::Vector<n,  x_T,Int>> x,
            mref<Tiny::Vector<m,  y_T,Int>> y
        )
        {
            if constexpr ( addto == Tensors::AddTo )
            {
                for( Int i = 0; i < m; ++i )
                {
                    y[i] += dot_buffers<n>( &A[i][0], x.data() );
                }
            }
            else
            {
                for( Int i = 0; i < m; ++i )
                {
                    y[i]  = dot_buffers<n>( &A[i][0], x.data() );
                }
            }
            
            // This is measurably slower.
//            fixed_dot_mm<m,1,n,addto>( &A[0][0], x.data(), y.data() );
        }
        
        template<int m, int n, typename A_T, typename x_T, IntQ Int>
        TOOLS_FORCE_INLINE Tiny::Vector<m,decltype(A_T(1) * x_T(1)),Int>
        Dot(
            cref<Tiny::Matrix<m,n,A_T,Int>> A,
            cref<Tiny::Vector<n,  x_T,Int>> x
        )
        {
            Tiny::Vector<m,decltype(A_T(1) * x_T(1)),Int> y;
            
            Dot<AddTo_T::False>(A,x,y);
            
            return y;
        }
        
        template<int m, int n, typename A_T, typename x_T, IntQ Int>
        TOOLS_FORCE_INLINE Tiny::Vector<m,decltype(A_T(1) * x_T(1)),Int>
        operator*(
             cref<Tiny::Matrix<m,n,A_T,Int>> A,
             cref<Tiny::Vector<n,  x_T,Int>> x
        )
        {
            return Dot(A,x);
        }
        
        
        
        template<typename Scal, IntQ Int>
        [[nodiscard]] TOOLS_FORCE_INLINE Scal Det_Kahan( cref<Tiny::Matrix<2,2,Scal,Int>> A )
        {
            return Det2D_Kahan( &A[0][0] );
        }
        
        
        template<int m, int n, typename Scal, IntQ Int>
        void OrthogonalizeRows( mref<Tiny::Matrix<m,n,Scal,Int>> A )
        {
            static_assert(m <= n, "");
            
            if( m <= 0 )
            {
                return;
            }
            
            normalize<n>( &A[0][0] );
            
            for( Int i = 1; i < m; ++i )
            {
                mptr<Scal> v_i = &A[i][0];
                for( Int j = 0; j < i; ++j )
                {
                    cptr<Scal> v_j = &A[j][0];

                    // v_i = v_i - v_j * innerprod( v_j, v_i );
                    
                    combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,n>(
                        -innerprod<n>(v_j,v_i), v_j, Scal(1), v_i
                    );
                }
                normalize<n>(v_i);
            }
        }
        
        template<int n, typename Scal, IntQ Int>
        [[nodiscard]] TOOLS_FORCE_INLINE
        Tiny::Matrix<n,n,Scal,Int> HouseholderReflector( cref<Vector<n,Scal,Int>> u )
        {
            Tiny::Matrix<n,n,Scal,Int> R;
            
            Scal uu = u.SquaredNorm();
            
            Scal factor = Frac<Scal>(Scal(2),uu);
            
            for( Int i = 0; i < n; ++i )
            {
                for( Int j = 0; j < n; ++j )
                {
                    R[i][j] = KroneckerDelta<Scalar::Real<Scal>>(i,j) - factor * u[i] * Conj(u[j]);
                }
            }
            
            return R;
        }
                
    } // namespace Tiny
    
    
    
} // namespace Tensors
