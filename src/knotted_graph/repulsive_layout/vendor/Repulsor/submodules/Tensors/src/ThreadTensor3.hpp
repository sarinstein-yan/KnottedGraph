#pragma once

namespace Tensors {

    template <typename Scal_, IntQ Int_, Size_T alignment = CacheLineWidth>
    class ThreadTensor3 final
    {
        using Scal = Scal_;
        using Real = typename Scalar::Real<Scal_>;
        using Int  = Int_;
        
        static constexpr Int    rank      = 3;
        static constexpr Size_T Alignment = alignment;
        
        using Tensor_T = Tensor2<Scal,Int,Alignment>;
        
    private:
        
        Int n = 0;
        std::array<Int,3> dims = {0,0,0};
        std::vector<Tensor_T> tensors;
        
    public:
        
        template<IntQ d0_T, IntQ d1_T, IntQ d2_T>
        ThreadTensor3( const d0_T d0, const d1_T d1, const d2_T d2 )
        :   n    { int_cast<Int>(ToSize_T(d0) * ToSize_T(d1) * ToSize_T(d2)) }
        ,   dims { int_cast<Int>(d0), int_cast<Int>(d1), int_cast<Int>(d2) }
        ,   tensors( std::vector<Tensor_T> ( ToSize_T(d0) ) )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)] = Tensor_T( dims[1], dims[2] );
                },
                thread_count
            );
        }
        
        template<IntQ d0_T, IntQ d1_T, IntQ d2_T>
        ThreadTensor3(
            const d0_T d0, const d1_T d1, const d2_T d2, cref<Scal> init
        )
        :   n    { int_cast<Int>(ToSize_T(d0) * ToSize_T(d1) * ToSize_T(d2)) }
        ,   dims { int_cast<Int>(d0), int_cast<Int>(d1), int_cast<Int>(d2) }
        ,   tensors( std::vector<Tensor_T> ( ToSize_T(d0) ) )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)] = Tensor_T( dims[1], dims[2], init );
                },
                thread_count
            );
        }
        
        template<typename S, IntQ d0_T, IntQ d1_T, IntQ d2_T>
        ThreadTensor3( cptr<S> a_, const d0_T d0, const d1_T d1, const d2_T d2 )
        :   ThreadTensor3( d0, d1, d2 )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)].Read( a_ + thread * dims[1] * dims[2]);
                },
                thread_count
            );
        }
        
        // Default constructor
        ThreadTensor3() = default;
        
        // Destructor
        ~ThreadTensor3(){
#ifdef TENSORS_BOUND_CHECKS
            print("~"+ClassName()+" { " + ToString(dims[0]) + ", " + ToString(dims[1]) + " }" );
#endif
        }

        // Copy constructor
        explicit ThreadTensor3( const ThreadTensor3 & other )
        :   ThreadTensor3( other.dims[0], other.dims[1], other.dims[2] )
        {
            print(ClassName()+" copy constructor");

            const Int thread_count = dims[0];

            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)].Read( other[thread].data() );
                },
                thread_count
            );
        }
        
        // Copy-cast constructor
        template<typename S, IntQ J, Size_T alignment_>
        explicit ThreadTensor3( const ThreadTensor3<S,J,alignment_> & other )
        :   ThreadTensor3( other.dims[0], other.dims[1], other.dims[2] )
        {
            print(ClassName()+" copy constructor");
            
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)].Read( other[ToSize_T(thread)].data() );
                },
                thread_count
            );
        }
        
        friend void swap( ThreadTensor3 & A, ThreadTensor3 & B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
#ifdef TENSORS_BOUND_CHECKS
            print(ClassName()+" swap");
#endif
            swap(A.tensors, B.tensors);
            swap(A.dims[0], B.dims[0]);
            swap(A.dims[1], B.dims[1]);
            swap(A.dims[2], B.dims[2]);
            swap(A.n , B.n );
        }
        
        // Copy constructor
        ThreadTensor3 & operator=(ThreadTensor3 B)
        {
            // see https://stackoverflow.com/a/3279550/8248900 for details
#ifdef TENSORS_BOUND_CHECKS
            print(ClassName()+" copy-and-swap");
#endif
            swap(*this, B);

            return *this;
            
        }
        
        // Move constructor
        ThreadTensor3( ThreadTensor3 && other ) noexcept
        :   ThreadTensor3()
        {
#ifdef TENSORS_BOUND_CHECKS
            print(ClassName()+" move constructor");
#endif
            swap(*this, other);
        }

        
        
        static constexpr Int Rank()
        {
            return Int(3);
        }

        
        void BoundCheck( const Int i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
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
        
        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i )
        {
            BoundCheck(i);

            return tensors[ToSize_T(i)].data();
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i ) const
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)].data();
        }

        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i, const Int j)
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)].data(j);
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i, const Int j) const
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)].data(j);
        }
        
        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i, const Int j, const Int k)
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)].data(j,k);
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i, const Int j, const Int k) const
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)].data(j,k);
        }

        TOOLS_FORCE_INLINE mref<Scal> operator()( const Int i, const Int j, const Int k)
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)](j,k);
        }
    
        TOOLS_FORCE_INLINE cref<Scal> operator()( const Int i, const Int j, const Int k) const
        {
            BoundCheck(i);
            
            return tensors[ToSize_T(i)](j,k);
        }
        
        void Fill( cref<Scal> init )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [this,&init]( const Int thread )
                {
                    tensors[ToSize_T(thread)].fill( init );
                },
                thread_count
            );
        }
        
        void SetZero()
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)].SetZero();
                },
                thread_count
            );
        }

        void Write( mptr<Scal> b ) const
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[ToSize_T(thread)].Write( b + dims[1] * dims[2] * thread );
                },
                thread_count
            );
        }
        
        template<typename S>
        void Write( const Int i, mptr<S> b ) const
        {
            tensors[ToSize_T(i)].Write( b );
        }
        
        template<typename S>
        void Write( const Int i, const Int j, mptr<S> b ) const
        {
            tensors[ToSize_T(i)].Write( j, b );
        }
        
        template<typename S>
        void Read( const Int i, cptr<S> b )
        {
            tensors[ToSize_T(i)].Read( b );
        }
        
        template<typename S>
        void Read( const Int i, const Int j, cptr<S> b )
        {
            tensors[ToSize_T(i)].Read( j, b );
        }
        
    public:
        
        TOOLS_FORCE_INLINE const Int * Dims() const
        {
            return &dims[0];
        }
        
        TOOLS_FORCE_INLINE const Int * Dimensions() const
        {
            return Dims();
        }
        
        TOOLS_FORCE_INLINE Int Dim( const Int i ) const
        {
            return i < Rank() ? dims[i] : Int(0);
        }
        
        TOOLS_FORCE_INLINE Int Dimension( const Int i ) const
        {
            return Dim(i);
        }
 
        Int Size() const
        {
            return n;
        }
        
        template<typename S, typename J>
        Tensor2<S,J> AddReduce() const
        {
            Tensor2<S,J> B ( dims[1], dims[2] );
            
            AddReduce( B.data(), false );
             
            return B;
        }
        
        template<typename S, IntQ J>
        void AddReduce( mref<Tensor2<S,J>> B, const bool add_to ) const
        {
            AddReduce( B.data(), add_to );
        }
        
        template<typename S>
        void AddReduce( mptr<S> B, const bool add_to ) const
        {
            if( add_to )
            {
                for( Int i = 0; i < dims[0]; ++ i )
                {
                    tensors[ToSize_T(i)].AddTo( B );
                }
            }
            else
            {
                // Write first slice.
                tensors[0].Write(B);
                
                for( Int i = 1; i < dims[0]; ++i )
                {
                    tensors[ToSize_T(i)].AddTo( B );
                }
            }
        }
        
        Int CountNaNs() const
        {
            Int counter = 0;
            for( Int i = 0 ; i < dims[0]; ++i )
            {
                counter += tensors[ToSize_T(i)].CountNaNs();
            }
            return counter;
        }
        
        
        mref<Tensor_T> operator[]( const Int i )
        {
            return tensors[ToSize_T(i)];
        }
        
        cref<Tensor_T> operator[]( const Int i ) const
        {
            return tensors[ToSize_T(i)];
        }
        
    public:
        
        static std::string ClassName()
        {
            return ct_string("ThreadTensor3")
                + "<" + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + to_ct_string(Alignment)
                + ">";
        }
        
    }; // ThreadTensor3
    
    
#ifdef LTEMPLATE_H

    
    template<FloatQ Scal, IntQ Int>
    inline mma::TensorRef<mreal> to_MTensorRef( cref<ThreadTensor3<Scal,Int>> A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dims());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        
        const Int size_ = A.Dim(1) * A.Dim(2);
        
        const Int thread_count = A.Dim(0);
        
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            A[ToSize_T(thread)].Write( &B.data()[size_ * thread] );
        }
        
        return B;
    }
    
    template<IntQ J, IntQ Int>
    inline mma::TensorRef<mint> to_MTensorRef( cref<ThreadTensor3<J,Int>> A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dims());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        
        const Int size_ = A.Dim(1) * A.Dim(2);
        
        const Int thread_count = A.Dim(0);
        
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            A[ToSize_T(thread)].Write( &B.data()[size_ * thread] );
        }
        
        return B;
    }
    
#endif
    
} // namespace Tensors
