#pragma once

namespace Tensors
{

#define TENSOR_T Tensor2

    template <typename Scal_, IntQ Int_, Size_T alignment = DefaultAlignment>
    class Tensor2 final
    {
    public:

        static constexpr Int_ rank = 2;
        
#include "Tensor_Common.hpp"
#include "Tensor_Common23.hpp"
        
    public:
        
        template<IntQ d0_T, IntQ d1_T>
        Tensor2( const d0_T d0, const d1_T d1)
        :   n    { int_cast<Int>(ToSize_T(d0) * ToSize_T(d1)) }
        ,   dims { int_cast<Int>(d0), int_cast<Int>(d1) }
        {
#ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " constructor (size = " + ToString(Size()) + ")");
#endif
            allocate();
        }
        
        template<IntQ d0_T, IntQ d1_T>
        Tensor2( const d0_T d0, const d1_T d1, cref<Scal> init )
        :   Tensor2( d0, d1 )
        {
            Fill( init );
        }
        
        template<typename S, IntQ d0_T, IntQ d1_T>
        Tensor2( cptr<S> a_, const d0_T d0, const d1_T d1 )
        :   Tensor2( d0, d1 )
        {
            Read(a_);
        }
        
    public:
        
        template<bool row_first = false, typename S>
        void WriteTransposed( mptr<S> b )
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            if constexpr ( row_first )
            {
                for( Int i = 0; i < d_0; ++i )
                {
                    for(Int j = 0; j < d_1; ++j )
                    {
                        b[d_0 * j + i] = static_cast<S>( a[d_1 * i + j ] );
                    }
                }
            }
            else
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    for( Int i = 0; i < d_0; ++i )
                    {
                        b[d_0 * j + i] = static_cast<S>( a[d_1 * i + j ] );
                    }
                }
            }
        }
        
        template<bool row_first = false, typename S>
        void ReadTransposed( cptr<S> b )
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
              
            if constexpr ( row_first )
            {
                for( Int i = 0; i < d_0; ++i )
                {
                    for(Int j = 0; j < d_1; ++j )
                    {
                        a[d_1 * i + j ] = static_cast<Scal>( b[ d_0 * j + i ] );
                    }
                }
            }
            else
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    for( Int i = 0; i < d_0; ++i )
                    {
                        a[d_1 * i + j ] = static_cast<Scal>( b[ d_0 * j + i ] );
                    }
                }
            }
        }
        
        // row-wise Write
        template< typename S>
        void Write( const Int i, mptr<S> b ) const
        {
            copy_buffer( data(i), b, dims[1] );
        }
        
        // row-wise Read
        template< typename S>
        void Read( const Int i, cptr<S> b )
        {
            copy_buffer( b, data(i), dims[1] );
        }
        
        
        
        template<Op op = Op::Id, typename S>
        void Read( cptr<S> B, const Int ldB, const Int thread_count = 1 ) const
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            if constexpr ( op == Op::Id )
            {
                Do<Parallel>(
                    [=,this]( const Int i )
                    {
                        copy_buffer( &B[ldB * i], &a[d_1 * i], d_1 );
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Conj )
            {
                Do<Parallel>(
                    [=,this]( const Int i )
                    {
                        for( Int j = 0; j < d_1; ++j )
                        {
                            a[d_1 * i + j] = scalar_cast<Scal>(Conj(B[ldB * i + j]));
                        }
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Trans )
            {
                Do<Parallel>(
                    [=,this]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            a[d_1 * i + j] = scalar_cast<Scal>(B[ldB * j + i]);
                        }
                    },
                    d_1, thread_count
                );
            }
            else if constexpr ( op == Op::ConjTrans )
            {
                Do<Parallel>(
                    [=,this]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            a[d_1 * i + j] = scalar_cast<Scal>(Conj(B[ldB * j + i]));
                        }
                    },
                    d_1, thread_count
                );
            }
            else
            {
                eprint(ClassName()+"::Write: No implementation for op available.");
            }
        }
        
//        template<typename S>
//        void Read( cptr<S> a_, const Int lda, const Int thread_count = 1 )
//        {
//            const Int d_0 = dims[0];
//            const Int d_1 = dims[1];
//            
//            Do<Parallel>(
//                [a_,lda,d_1,this]( const Int i )
//                {
//                    copy_buffer( &a_[lda * i], &a[d_1 * i], d_1 );
//                },
//                d_0, thread_count
//            );
//        }
//        
//        template<typename S>
//        void Write( mptr<S> a_, const Int lda, const Int thread_count = 1 )
//        {
//            const Int d_0 = dims[0];
//            const Int d_1 = dims[1];
//            
//            Do<Parallel>(
//                [=,this]( const Int i )
//                {
//                    copy_buffer( &a[d_1 * i], &a_[lda * i], d_1 );
//                },
//                d_0, thread_count
//            );
//        }
        
        template<Op op = Op::Id, typename S>
        void Write( mptr<S> B, const Int ldB, const Int thread_count = 1 ) const
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            if constexpr ( op == Op::Id )
            {
                Do<Parallel>(
                    [=,this]( const Int i )
                    {
                        copy_buffer( &a[d_1 * i], &B[ldB * i], d_1 );
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Conj )
            {
                Do<Parallel>(
                    [=,this]( const Int i )
                    {
                        for( Int j = 0; j < d_1; ++j )
                        {
                            B[ldB * i + j] = scalar_cast<S>(Conj(a[d_1 * i + j]));
                        }
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Trans )
            {
                Do<Parallel>(
                    [=,this]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            B[ldB * j + i] = scalar_cast<S>(a[d_1 * i + j]);
                        }
                    },
                    d_1, thread_count
                );
            }
            else if constexpr ( op == Op::ConjTrans )
            {
                Do<Parallel>(
                    [=,this]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            B[ldB * j + i] = scalar_cast<S>(Conj(a[d_1 * i + j]));
                        }
                    },
                    d_1, thread_count
                );
            }
            else
            {
                eprint(ClassName()+"::Write: No implementation for op available.");
            }
        }
        void SetIdentity( const Int thread_count = 1)
        {
            Scal * A = a;
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            Do<Parallel>(
                [A,d_0,d_1]( const Int i )
                {
                    if( i > Int(0) )
                    {
                        zerofy_buffer(&A[0],i-Int(1));
                    }
                    A[i] = Int(1);
                    if( i < d_1 )
                    {
                        zerofy_buffer(&A[i+1],d_1-i);
                    }
                },
                d_0, thread_count
            );
        }

    private:
        
        void BoundCheck( const Int i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            if( a == nullptr )
            {
                eprint(ClassName() + ": pointer is nullptr.");
            }
            if( std::cmp_less(i,Int(0)) || std::cmp_greater(i,dims[0]) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
#else
            (void)i;
#endif
        }
        
        void BoundCheck( const Int i, const Int j ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            if( a == nullptr )
            {
                eprint(ClassName() + ": pointer is nullptr.");
            }
            if( std::cmp_less(i,Int(0)) || std::cmp_greater(i,dims[0]) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
            
            if( std::cmp_less(j,Int(0)) || std::cmp_greater(j,dims[1]) )
            {
                eprint(ClassName()+": second index " + ToString(j) + " is out of bounds [ 0, " + ToString(dims[1]) +" [.");
            }
#else
            (void)i;
            (void)j;
#endif
        }
        
    public:

        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i ) noexcept
        {
            BoundCheck(i);
            
            return &a[i * dims[1]];
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i ) const noexcept
        {
            BoundCheck(i);
            
            return &a[i * dims[1]];
        }
        
        TOOLS_FORCE_INLINE mref<Scal> operator()(const Int i, const Int j ) noexcept
        {
            BoundCheck(i,j);
            
            return a[ i * dims[1] + j];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> operator()( const Int i, const Int j ) const noexcept
        {
            BoundCheck(i,j);
            
            return a[i * dims[1] + j];
        }
        
        TOOLS_FORCE_INLINE mptr<Scal> operator[](const Int i) noexcept
        {
            BoundCheck(i);
            
            return data(i);
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> operator[](const Int i) const noexcept
        {
            BoundCheck(i);
            
            return data(i);
        }
        
        
        
        auto WriteAccess()
        {
            return [this]( const Int i, const Int j ) -> Scal&
            {
                BoundCheck(i,j);
                
                return a[i * dims[1] + j];
            };
        }
        
        auto ReadAccess() const
        {
            return [this]( const Int i, const Int j ) -> Scal
            {
                BoundCheck(i,j);
                
                return a[i * dims[1] + j];
            };
        }
        
        
        inline friend std::ostream & operator<<( std::ostream & s, cref<TENSOR_T> tensor )
        {
            return s << OutString::FromMatrix(tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1));
        }

        inline friend std::string ToString( cref<TENSOR_T> tensor )
        {
            return OutString::FromMatrix(tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1));
        }
        
        inline friend std::string ToString( cref<TENSOR_T> tensor, cref<std::string> line_prefix )
        {
            return OutString::FromArray(
                tensor.ReadAccess(),
                tensor.Dim(0), line_prefix + "{\n", ",\n", "\n" + line_prefix + "}",
                tensor.Dim(1), line_prefix + " { ", ", ", " }"
            );
        }
        
    public:
        
        // Typically, one wants to keep the data, so copy = true is default.
        template< bool copyQ>
        void Resize( const Int d_0_, const Int d_1_, const Int thread_count = 1 )
        {
            const Int d_0 = Tools::Ramp(d_0_);
            const Int d_1 = Tools::Ramp(d_1_);
            
            Tensor2 b ( d_0, d_1 );
            
            if constexpr ( copyQ )
            {
                const Int row_count = Tools::Min( b.Dim(0), dims[0] );
                const Int col_count = Tools::Min( b.Dim(1), dims[1] );
                
                const cptr<Scal> X = this->data();
                const mptr<Scal> Y = b.data();
                
                const Int ldX = Dim(1);
                const Int ldY = b.Dim(1);
                
                copy_matrix<VarSize,VarSize,Parallel>(
                    X, ldX, Y, ldY, row_count, col_count, thread_count
                );
            }
            
            swap( *this, b );
        }
        
        template< bool copyQ>
        void RequireSize( const Int d_0, const Int d_1, const Int thread_count = 1 )
        {
            if( (dims[0] < d_0) || (dims[1] < d_1) )
            {
                Resize<copyQ>(d_0,d_1,thread_count);
            }
        }
        
        template<typename FillFun_T>
        void FillByFunction( FillFun_T && f )
        {
            for( Int i = 0; i < Dim(0); ++i )
            {
                for( Int j = 0; j < Dim(1); ++j )
                {
                    this->operator()(i,j) = f(i,j);
                }
            }
        }
        
    public:
        
        static std::string ClassName() noexcept
        {
            return ct_string("Tensor2")
                + "<" + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + to_ct_string(alignment)
                + ">";
        }
        
    }; // Tensor2

    template<typename Scal, IntQ Int, typename S>
    Tensor2<Scal,Int> ToTensor2( cptr<S> a_, const Int d0, const Int d1, const bool transpose = false )
    {
        Tensor2<Scal,Int> result ( d0, d1 );

        if( transpose )
        {
            result.ReadTransposed(a_);
        }
        else
        {
            result.Read(a_);
        }
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    
    template<typename Scal, IntQ Int>
    Tensor2<Scal,Int> from_MatrixRef( cref<mma::TensorRef<mreal>> A )
    {
        return ToTensor2<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    
    template<typename Scal, IntQ Int>
    Tensor2<Scal,Int> from_MatrixRef( cref<mma::TensorRef<mint>> A )
    {
        return ToTensor2<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    

    template<typename Scal, IntQ Int, 
        class = typename std::enable_if_t<FloatQ<Scal>>
    >
    mma::MatrixRef<mreal> to_transposed_MTensorRef( cref<Tensor2<Scal,Int>> B )
    {
        Int rows = B.Dim(0);
        Int cols = B.Dim(1);
        auto A = mma::makeMatrix<mreal>( cols, rows );

        cptr<double> a_out = A.data();

        for( Int i = 0; i < rows; ++i )
        {
            for( Int j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mreal>( B(i,j) );
            }
        }

        return A;
    }
    
    template<IntQ J, IntQ Int>
    mma::MatrixRef<mint> to_transposed_MTensorRef( cref<Tensor2<J,Int>> B )
    {
        Int rows = B.Dim(0);
        Int cols = B.Dim(1);
        auto A = mma::makeMatrix<mint>( cols, rows );

        cptr<double> a_out = A.data();

        for( Int i = 0; i < rows; ++i )
        {
            for( Int j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mint>( B(i,j) );
            }
        }

        return A;
    }

    
#endif

    // Should be only a fall-back. BLAS is _much_ faster.
    template<typename Scal, IntQ I1, IntQ I2, IntQ I3>
    void Dot(
        cref<Tensor2<Scal,I1>> A,
        cref<Tensor1<Scal,I2>> x,
        mref<Tensor1<Scal,I3>> y
    )
    {
        // TODO: Use BLAS if available.
        
        I3 m = A.Dim(0);
        I3 n = Min(A.Dim(1),x.Dim(0));

        if( y.Dim(0) != m )
        {
            y = Tensor1<Scal,I3> ( m);
        }
        
        for( I3 i = 0; i < m; ++i )
        {
            Scal y_i = A(i,0) * x[0];
            
            for( I3 j = 1; j < n; ++j )
            {
                y_i += A(i,j) * x[j];
            }
            
            y[i] = y_i;
        }
    }
    
    // Should be only a fall-back. BLAS is _much_ faster.
    template<typename Scal, IntQ I1, IntQ I2, IntQ I3>
    void Dot(
        cref<Tensor1<Scal,I1>> x,
        cref<Tensor2<Scal,I2>> A,
        mref<Tensor1<Scal,I3>> y
    )
    {
        // TODO: Use BLAS if available.
        
        I3 m = Min(A.Dim(0),x.Dim(0));
        I3 n = A.Dim(1);
        
        if( y.Dim(0) != n )
        {
            y = Tensor1<Scal,I3> (n);
        }

        {
            const Scal x_0 = x[0];
            for( I3 j = 0; j < n; ++j )
            {
                y[j] = x_0 * A(0,j);
            }
        }
        
        for( I3 i = 1; i < m; ++i )
        {
            const Scal x_i = x[i];
            
            for( I3 j = 0; j < n; ++j )
            {
                y[j] += x_i * A(i,j);
            }
        }
    }
    
    template<
        typename A_T, typename B_T, IntQ Int,
        typename C_T = decltype( A_T(1) * B_T(1))
    >
    Tensor2<C_T,Int> KroneckerProduct(
        cref<Tensor2<A_T,Int>> A, cref<Tensor2<B_T,Int>> B, const Int thread_count
    )
    {
        const Int mA = A.Dim(0);
        const Int nA = A.Dim(1);
        const Int mB = B.Dim(0);
        const Int nB = B.Dim(1);

        const Int mC = mA * mB;
        const Int nC = nA * nB;

        Tensor2<C_T,Int> C ( mC, nC );

        cptr<A_T> a = A.data();
        cptr<B_T> b = B.data();
        mptr<C_T> c = C.data();

        Do<Parallel>(
            [=](const Int i )
            {
                const Int mAi = mA * i;
                const Int mBi = mB * i;

                for( Int k = 0; k < mB; ++k )
                {
                    const Int nBk = nB * k;
                    const Int mCI = mC * (mBi + k);
                    
                    outerprod_buffers( &a[mAi], &b[nBk], &c[mB * i + k] );
               }
           },
           mA, thread_count
        );
               
        return C;
    }
    
    
    template<typename Scal, IntQ Int>
    OutString ToStringTSV( cref<Tensor2<Scal,Int>> X )
    {
//        return MatrixStringTSV( ToSize_T(X.Dim(0)), ToSize_T(X.Dim(1)), X.data(), ToSize_T(X.Dim(1)) );
        
        return OutString::FromMatrix<Format::Matrix::TSV>( X.data(), X.Dim(0), X.Dim(1) );
    }


    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
    
} // namespace Tensors
