#pragma once

namespace Tensors
{

#define TENSOR_T Tensor1 

    template <typename Scal_, IntQ Int_, Size_T alignment = DefaultAlignment>
    class Tensor1 final
    {
    public:

        static constexpr Int_ rank = 1;
    
#include "Tensor_Common.hpp"
        
    public:
        
        template<IntQ d0_T>
        explicit Tensor1( const d0_T d0 )
        :   n    { int_cast<Int>(d0) }  // Using int_cast to get error messages.
        {
#ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " constructor (size = " + ToString(Size()) + ")");
#endif
            allocate();
        }
        
        template<IntQ d0_T>
        Tensor1( const d0_T d0, cref<Scal> init )
        :   Tensor1( d0 ) // Using int_cast to get error messages.
        {
            Fill( init );
        }
        
        template<typename S, IntQ d0_T>
        Tensor1( cptr<S> a_, const d0_T d0 )
        :   Tensor1( d0 )
        {
            Read(a_);
        }
        
        // Copy constructor
        TENSOR_T( const TENSOR_T & other )
        :   n    ( other.n    )
        {
        #ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " copy-constructor (size = " + ToString(other.Size()) + ")");
        #endif
            allocate();
            Read(other.a);
        }

        // Copy-cast constructor
        template<typename S, IntQ J, Size_T alignment_>
        explicit TENSOR_T( const TENSOR_T<S,J,alignment_> & other )
        :   n    ( other.Size()    )
        {
        #ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " copy-cast constuctor (size = " + ToString(other.Size()) + ")");
        #endif
            allocate();
            Read(other.data());
        }

        inline friend void swap( TENSOR_T & A, TENSOR_T & B) noexcept
        {
        #ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " swap (sizes = {" + ToString(A.Size()) + "," + ToString(B.Size()) + "})");
        #endif
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
            
            if( &A == &B )
            {
                wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
            }
            else
            {
                swap( A.a   , B.a       );
                swap( A.n   , B.n       );
            }
        }

        // Copy assignment operator
        // We ship our own because we can do a few optimizations here.
        mref<TENSOR_T> operator=( const TENSOR_T & other )
        {
        #ifdef TENSORS_ALLOCATION_LOGS
                logprint(ClassName() + " copy-assignment (size = " + ToString(other.Size()) + ")");
        #endif
            if( this != &other )
            {
                if( n != other.n )
                {
                    n = other.n;

#ifdef TENSORS_ALLOCATION_LOGS
                    logprint( ClassName() + " reallocation (size = " + ToString(other.Size()) + ")");
        #endif
                    
                    safe_free(a);
                    allocate();
                }
                Read( other.a );
            }
            return *this;
        }
        
        
        // Copy-cast-assignment operator
        template<
            typename S, IntQ J, Size_T alignment_,
            class = std::enable_if_t<(!SameQ<S,Scal>) || (!SameQ<J,Int>) || ( alignment_ != Alignment)>
        >
        mref<TENSOR_T> operator=( const TENSOR_T<S,J,alignment_> & other )
        {

        #ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " copy-cast-assignment (size = " + ToString(other.Size()) + ")");
        #endif

            if( std::cmp_not_equal( Dim(0), other.Dim(0) ) )
            {
        #ifdef TENSORS_ALLOCATION_LOGS
                logprint(ClassName() + " reallocation (size = " + ToString(other.Size()) + ")");
        #endif
                n = other.Dim(0);
                safe_free(a);
                allocate();
            }
            Read( other.data() );
            
            return *this;
        }
        
    private:
        
        template<IntQ I>
        void BoundCheck( const I i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            if( a == nullptr )
            {
                eprint(ClassName()+": pointer is nullptr.");
            }
            if( std::cmp_less(i,Int(0)) || std::cmp_greater(i,n) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(n) +" [.");
            }
#else
            (void)i;
#endif
        }
        
    public:
        
        TOOLS_FORCE_INLINE cptr<Int> Dims() const noexcept
        {
            return &n;
        }

        TOOLS_FORCE_INLINE Int Dim( const Int i ) const noexcept
        {
            return ( i == Int(0) ) ? n : Scalar::Zero<Int>;
        }
        
        template<IntQ I>
        TOOLS_FORCE_INLINE mptr<Scal> data( const I i ) noexcept
        {
            BoundCheck(i);
            return &a[i];
        }
        
        template<IntQ I>
        TOOLS_FORCE_INLINE cptr<Scal> data( const I i ) const noexcept
        {
            BoundCheck(i);
            return &a[i];
        }
        
        template<IntQ I>
        TOOLS_FORCE_INLINE mref<Scal> operator()(const I i) noexcept
        {
            BoundCheck(i);
            return a[i];
        }
        
        template<IntQ I>
        TOOLS_FORCE_INLINE cref<Scal> operator()(const I i) const noexcept
        {
            BoundCheck(i);
            return a[i];
        }
        
        template<IntQ I>
        TOOLS_FORCE_INLINE mref<Scal> operator[](const I i) noexcept
        {
            BoundCheck(i);
            return a[i];
        }
        
        template<IntQ I>
        TOOLS_FORCE_INLINE cref<Scal> operator[](const I i) const noexcept
        {
            BoundCheck(i);
            return a[i];
        }
        

        
        TOOLS_FORCE_INLINE mref<Scal> First() noexcept
        {
            return a[0];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> First() const noexcept
        {
            return a[0];
        }

        TOOLS_FORCE_INLINE mref<Scal> Last() noexcept
        {
            return a[n-1];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> Last() const noexcept
        {
            return a[n-1];
        }
        
        
        
        auto WriteAccess()
        {
            return [this]( const Int i ) -> Scal&
            {
                BoundCheck(i);
                return a[i];
            };
        }
        
        auto ReadAccess() const
        {
            return [this]( const Int i ) -> Scal
            {
                BoundCheck(i);
                return a[i];
            };
        }
        
        inline friend std::ostream & operator<<( std::ostream & s, cref<TENSOR_T> tensor )
        {
            return s << OutString::FromVector(tensor.ReadAccess(), tensor.Dim(0));
        }

        inline friend std::string ToString( cref<TENSOR_T> tensor )
        {
            return OutString::FromVector(tensor.ReadAccess(), tensor.Dim(0));
        }
        
        inline friend std::string ToString( cref<TENSOR_T> tensor, cref<std::string> line_prefix )
        {
            return OutString::FromArray(
                tensor.ReadAccess(),
                tensor.Dim(0), line_prefix + "{ ", ", ", " }"
            );
        }
        
        
        
        template<bool copy>
        void Resize( const Int m_, const Int thread_count = 1 )
        {
            const Int m = Ramp(m_);
            
            TENSOR_T b (m);
            
            if constexpr ( copy )
            {
                if( m <= n )
                {
                    b.ReadParallel(a,thread_count);
                }
                else
                {
                    WriteParallel(b.data(),thread_count);
                }
            }
            
            swap( *this, b );
        }
        
        template<bool copy>
        void RequireSize( const Int m, const Int thread_count = 1 )
        {
            if( m > n )
            {
                Resize<copy>( m, thread_count );
            }
        }

        void Accumulate()
        {
            Tools::Accumulate<Sequential>(a, n);
        }
        
        template<Parallel_T parQ = Parallel>
        void Accumulate( const Int thread_count )
        {
            Tools::Accumulate<parQ>(a, n, thread_count);
        }
        
        Scal Total(const Int thread_count = 1) const
        {
            return total_buffer<VarSize,Parallel>(a,n,thread_count);
        }
        
        inline friend Scal Total( cref<TENSOR_T> x )
        {
            return x.Total();
        }
        
        inline friend Scal Dot( cref<TENSOR_T> x, cref<TENSOR_T> y, const Int thread_count = 1 )
        {
            if( x.n != y.n )
            {
                eprint(ClassName()+"::Dot: Sizes of vectors differ. Doing nothing.");
                return 0;
            }
            
            return dot_buffers<VarSize,Parallel>(x.a,y.a,x.n,thread_count);
        }
        
        void iota( const Int thread_count = 1 )
        {
            iota_buffer( a, int_cast<Size_T>(n), int_cast<Size_T>(thread_count) );
        }
        
            
        template<typename FillFun_T>
        void FillByFunction( FillFun_T && f )
        {
            for( Int i = 0; i < Dim(0); ++i )
            {
                this->operator()(i) = f(i);
            }
        }
        
    public:
        
        [[nodiscard]] std::string friend ToString( cref<TENSOR_T> T, const Int i_begin, const Int i_end )
        {
            if( (i_begin >= 0) && ( i_end <= T.n) )
            {
                return OutString::FromVector(
                    &T.a[i_begin],
                    Tools::Max(int_cast<Size_T>(0),int_cast<Size_T>(i_end-i_begin))
                );
            }
            else
            {
                return OutString::FromVector(T.a,0);
            }
        }

    public:
        
        static std::string ClassName() noexcept
        {
            return ct_string("Tensor1")
                + "<" + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + to_ct_string(alignment)
                + ">";
        }
        
    }; // Tensor1
    
    template<typename Scal, IntQ Int, Size_T alignment = DefaultAlignment>
    Tensor1<Scal,Int,alignment> iota( const Int size_ )
    {
        Tensor1<Scal,Int,alignment> v (size_);
        
        v.iota();
        
        return v;
    }
    
//    template<typename Scal, IntQ Int, Size_T alignment = DefaultAlignment, typename S>
//    Tensor1<Scal,Int,alignment> ToTensor1( mptr<S> a_, const Int d0 )
//    {
//        Tensor1<Scal,Int,alignment> result (d0);
//
//        result.Read(a_);
//        
//        return result;
//    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, IntQ Int, Size_T alignment = DefaultAlignment>
    Tensor1<Scal,Int,alignment> from_VectorRef( cref<mma::TensorRef<mreal>> A )
    {
        return Tensor1<Scal,Int,alignment>( A.data(), A.dimensions()[0] );
    }
    
    template<typename Scal, IntQ Int, Size_T alignment = DefaultAlignment>
    Tensor1<Scal,Int,alignment> from_VectorRef( cref<mma::TensorRef<mint>> A )
    {
        return Tensor1<Scal,Int,alignment>( A.data(), A.dimensions()[0] );
    }
    
#endif
    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
