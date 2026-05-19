#pragma once

namespace Tensors
{

#define TENSOR_T Tensor3

    template <typename Scal_, IntQ Int_, Size_T alignment = DefaultAlignment>
    class TENSOR_T final
    {
    public:
        
        static constexpr Int_ rank = 3;
        
#include "Tensor_Common.hpp"
#include "Tensor_Common23.hpp"
        
    public:
        
        template<
        IntQ d0_T, IntQ d1_T, IntQ d2_T>
        Tensor3( const d0_T d0, const d1_T d1, const d2_T d2 )
        :   n    { int_cast<Int>(ToSize_T(d0) * ToSize_T(d1) * ToSize_T(d2)) }
        ,   dims { int_cast<Int>(d0), int_cast<Int>(d1), int_cast<Int>(d2) }
        {
#ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " constructor (size = " + ToString(Size()) + ")");
#endif
            allocate();
        }
        
        template<IntQ d0_T, IntQ d1_T, IntQ d2_T>
        Tensor3( const d0_T d0, const d1_T d1, const d2_T d2, cref<Scal> init )
        :   Tensor3( d0, d1, d2 )
        {
            Fill( init );
        }
        
        template<typename S, IntQ d0_T, IntQ d1_T, IntQ d2_T>
        Tensor3( cptr<S> a_, const d0_T d0, const d1_T d1, const d2_T d2 )
        :   Tensor3( d0, d1, d2 )
        {
            Read(a_);
        }
        
    protected:
        
        void BoundCheck( const Int i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            if( a == nullptr )
            {
                eprint(ClassName() + ": pointer is nullptr.");
            }
            if( std::cmp_less(i,Int(0)) || std::cmp_greater(i,dims[0]) )
            {
                eprint(ClassName() + ": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
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
        
        void BoundCheck( const Int i, const Int j, const Int k ) const
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
            if( std::cmp_less(k,Int(0)) || std::cmp_greater(k,dims[2]) )
            {
                eprint(ClassName()+": third index " + ToString(k) + " is out of bounds [ 0, " + ToString(dims[2]) +" [.");
            }
#else
            (void)i;
            (void)j;
            (void)k;
#endif
        }
        
    public:
        

        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i ) noexcept
        {
            BoundCheck(i);
            return &a[i * dims[1] * dims[2]];
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i ) const noexcept
        {
            BoundCheck(i);
            return &a[i * dims[1] * dims[2]];
        }

        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i, const Int j ) noexcept
        {
            BoundCheck(i,j);
            return &a[( i * dims[1] + j ) * dims[2]];
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i, const Int j ) const noexcept
        {
            BoundCheck(i,j);
            return &a[( i * dims[1] + j ) * dims[2]];
        }

        
        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i, const Int j, const Int k) noexcept
        {
            BoundCheck(i,j,k);
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        
        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i, const Int j, const Int k) const noexcept
        {
            BoundCheck(i,j,k);
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        TOOLS_FORCE_INLINE mref<Scal> operator()( const Int i, const Int j, const Int k) noexcept
        {
            BoundCheck(i,j,k);
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> operator()( const Int i, const Int j, const Int k) const noexcept
        {
            BoundCheck(i,j,k);
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        auto WriteAccess()
        {
            return [this]( const Int i, const Int j, const Int k ) -> Scal&
            {
                BoundCheck(i,j,k);
                return a[( i *  dims[1] + j ) * dims[2] + k];
            };
        }
        
        auto ReadAccess() const
        {
            return [this]( const Int i, const Int j, const Int k ) -> Scal
            {
                BoundCheck(i,j,k);
                return a[( i *  dims[1] + j ) * dims[2] + k];
            };
        }
        
        template< typename S>
        void Write( const Int i, mptr<S> b ) const
        {
            copy_buffer( data(i), b, dims[1] * dims[2] );
        }
        
        template< typename S>
        void Write( const Int i, const Int j, mptr<Scal> b ) const
        {
            copy_buffer( data(i,j), b, dims[2] );
        }
        
        template< typename S>
        void Read( const Int i, cptr<S> b )
        {
            copy_buffer( b, data(i), dims[1] * dims[2] );
        }
        
        template< typename S>
        void Read( const Int i, const Int j, cptr<S> b )
        {
            copy_buffer( b, data(i,j), dims[2] );
        }
        
        template<typename FillFun_T>
        void FillByFunction( FillFun_T && f )
        {
            for( Int i = 0; i < Dim(0); ++i )
            {
                for( Int j = 0; j < Dim(1); ++j )
                {
                    for( Int k = 0; k < Dim(2); ++k )
                    {
                        this->operator()(i,j,k) = f(i,j,k);
                    }
                }
            }
        }
        
        
    
        inline friend std::ostream & operator<<( std::ostream & s, cref<TENSOR_T> tensor )
        {
            return s << OutString::FromCube(
                tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1), tensor.Dim(2)
            );
        }

        inline friend std::string ToString( cref<TENSOR_T> tensor )
        {
            return OutString::FromCube(tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1), tensor.Dim(2));
        }
        
        inline friend std::string ToString( cref<TENSOR_T> tensor, cref<std::string> line_prefix )
        {
            return OutString::FromArray(
                tensor.ReadAccess(),
                tensor.Dim(0), line_prefix + "{\n", ",\n", "\n" + line_prefix + "}",
                tensor.Dim(1), line_prefix + " { ", ", ", " }",
                tensor.Dim(2),               "{ ", ", ", " }"
            );
        }
        
    public:
        
        static std::string ClassName() noexcept
        {
            return ct_string("Tensor3")
                + "<" + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + to_ct_string(alignment)
                + ">";
        }
        
    }; // Tensor3
    
    
//    template<typename Scal, IntQ Int, typename S>
//    Tensor3<Scal,Int> ToTensor3( cptr<S> a_, const Int d0, const Int d1, const Int d2 )
//    {
//        Tensor3<Scal,Int> result ( d0, d1, d2 );
//
//        result.Read(a_);
//        
//        return result;
//    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, IntQ Int>
    Tensor3<Scal,Int> from_CubeRef( cref<mma::TensorRef<mreal>> A )
    {
        return Tensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
    template<typename Scal, IntQ Int>
    Tensor3<Scal,Int> from_CubeRef( cref<mma::TensorRef<mint>> A )
    {
        return Tensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
#endif

#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
