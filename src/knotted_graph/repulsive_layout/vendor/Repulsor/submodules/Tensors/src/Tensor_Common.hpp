public:

    using Scal = Scal_;
    using Real = typename Scalar::Real<Scal_>;
    using Int  = Int_;

    static constexpr Size_T Alignment = alignment;

private:

    Scal * restrict a = nullptr ;
    Int n = 0;

public:

    // Default constructor
    TENSOR_T() = default;

    // Destructor
    ~TENSOR_T() noexcept
    {
#ifdef TENSORS_ALLOCATION_LOGS
        logprint(ClassName() + " destructor (size = " + ToString(Size()) + ")");
#endif
        deallocate();
    }

    // Move constructor
    TENSOR_T( TENSOR_T && other ) noexcept
    :   TENSOR_T()
    {
#ifdef TENSORS_ALLOCATION_LOGS
        logprint(ClassName() + " move-constructor (size = " + ToString(other.Size()) + ")");
#endif
        swap(*this, other);
    }

    // Move assignment operator
    mref<TENSOR_T> operator=( TENSOR_T && other ) noexcept
    {
#ifdef TENSORS_ALLOCATION_LOGS
        logprint(ClassName() + " move-assignment (size = " + ToString(other.Size()) + ")");
#endif
        if( this == &other )
        {
#ifdef TENSORS_ALLOCATION_LOGS
            wprint("An object of type " + ClassName() + " has been move-assigned to itself.");
#endif
        }
        else
        {
            swap( *this, other );
        }
        return *this;
    }

public:


    static constexpr Int Rank() noexcept
    {
        return static_cast<Int>(rank);
    }

    Int Size() const noexcept
    {
        return n;
    }

    template<typename S>
    void Read( cptr<S> a_ )
    {
        copy_buffer( a_, a, n );
    }

    //// Parallelized version.
    template<typename S>
    void ReadParallel( cptr<S> a_, const Int thread_count )
    {
        copy_buffer<VarSize,Parallel>( a_, a, n, thread_count );
    }

    template<typename R>
    std::enable_if_t<Scalar::ComplexQ<Scal> && !Scalar::ComplexQ<R>,void>
    Read( cptr<R> re, cptr<R> im )
    {
        for( Int i = 0; i < n; ++i )
        {
            a[i].real( re[i] );
            a[i].imag( im[i] );
        }
    }

    template<typename S>
    void Write( mptr<S> a_ ) const
    {
        copy_buffer( a, a_, static_cast<Size_T>(n) );
    }

    template<typename S>
    void WriteParallel( mptr<S> a_, const Int thread_count ) const
    {
        copy_buffer<VarSize,Parallel>( a, a_, n, thread_count );
    }

    template<typename R>
    std::enable_if_t<Scalar::ComplexQ<Scal> && !Scalar::ComplexQ<R>,void>
    Write( mptr<R> re, mptr<R> im ) const
    {
        for( Int i = 0; i < n; ++i )
        {
            re[i] = real(a[i]);
            im[i] = imag(a[i]);
        }
    }

    void Fill( cref<Scal> init )
    {
        fill_buffer( a, init, static_cast<Size_T>(n) );
    }

    void Fill( cref<Scal> init, const Size_T thread_count )
    {
        fill_buffer<VarSize,Parallel>( a, init, n, thread_count );
    }

    void SetZero()
    {
        zerofy_buffer( a, n );
    }

    void SetZero( const Int thread_count )
    {
        zerofy_buffer<VarSize,Parallel>( a, n, thread_count );
    }

    // TODO: Have to make the pseudorandom number generator configurable.
    void Randomize( const Int thread_count = 1 )
    {
        static_assert( Scalar::FloatQ<Scal>, "" );
        
        TOOLS_PTIMER(timer,ClassName()+"::Randomize");
        // This uses std::mt19937_64.
        // Moreover, the pseudorandom number generators are initilized per call.
        // So this is not very efficient.
        
        using SD_T = std::random_device;
        using MT_T = std::mt19937_64;
        
        using SD_UInt = SD_T::result_type;
        using MT_UInt = MT_T::result_type;
        
        constexpr Size_T seed_size = (MT_T::state_size * sizeof(MT_UInt)) / sizeof(SD_UInt);
        
        std::vector<SD_UInt> seed_array ( seed_size );
        
        std::vector<MT_T> engines;
        
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            std::generate( seed_array.begin(), seed_array.end(), SD_T() );
        
            std::seed_seq seed ( seed_array.begin(), seed_array.end() );
        
            engines.emplace_back( seed );
        }
        
        if constexpr (Scalar::RealQ<Scal> )
        {
            ParallelDo(
                [&,this]( const Int thread )
                {
                    const Int i_begin = JobPointer( n, thread_count, thread     );
                    const Int i_end   = JobPointer( n, thread_count, thread + 1 );
                    
                    MT_T & engine = engines[ToSize_T(thread)];
                    
                    std::uniform_real_distribution<Real> unif(-Scalar::One<Real>,Scalar::One<Real>);
                   
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        a[i] = unif(engine);
                    }
                },
                thread_count
            );
        }
        else
        {
            ParallelDo(
               [&,this]( const Int thread )
               {
                   const Int i_begin = JobPointer( n, thread_count, thread     );
                   const Int i_end   = JobPointer( n, thread_count, thread + 1 );
                   
                   MT_T & engine = engines[ToSize_T(thread)];
                   
                   std::uniform_real_distribution<Real> unif(-Scalar::One<Real>,Scalar::One<Real>);
                   
                   for( Int i = i_begin; i < i_end; ++i )
                   {
                       a[i] = Scal( unif(engine), unif(engine) );
                   }
               },
               thread_count
           );
        }
    }


    // Compatibility layer to older versions.
    void Random( const Int thread_count = 1 )
    {
        Randomize(thread_count);
    }

private:

    TOOLS_FORCE_INLINE void allocate() noexcept
    {
        safe_alloc( a, ToSize_T(n), Alignment );
        
//        std::allocator<Scal> allocator;
//        a = allocator.allocate(ToSize_T(n));
    }

    TOOLS_FORCE_INLINE void deallocate() noexcept
    {
        safe_free(a);

//        std::allocator<Scal> allocator;
//        allocator.deallocate(a,ToSize_T(n));
    }


public:

    TOOLS_FORCE_INLINE mptr<Scal> begin() noexcept
    {
        return a;
    }

    TOOLS_FORCE_INLINE cptr<Scal> begin() const noexcept
    {
        return a;
    }

    TOOLS_FORCE_INLINE mptr<Scal> end() noexcept
    {
        return &a[n];
    }

    TOOLS_FORCE_INLINE cptr<Scal> end() const noexcept
    {
        return &a[n];
    }

    TOOLS_FORCE_INLINE cptr<Int> Dimensions() const noexcept
    {
        return Dims();
    }

    TOOLS_FORCE_INLINE Int Dimension( const Int i ) const
    {
        return Dim(i);
    }

public:

    TOOLS_FORCE_INLINE mptr<Scal> data() noexcept
    {
        return a;
    }

    TOOLS_FORCE_INLINE cptr<Scal> data() const noexcept
    {
        return a;
    }


    void AddFrom( cptr<Scal> b )
    {
        add_to_buffer( b, a, n );
    }

    void AddTo( mptr<Scal> b ) const
    {
        add_to_buffer( a, b, n );
    }

    Int CountNaNs() const
    {
        Int counter = 0;
        
        for( Int i = 0 ; i < n; ++i )
        {
            counter += Int(NaNQ(a[i]));
        }
        
        return counter;
    }


    template <typename Dummy = Scal>
    std::enable_if_t<SameQ<Real,Dummy>,std::pair<Real,Real>> MinMax( Int thread_count = 1 ) const
    {
        return minmax_buffer( a, n, thread_count );
    }

    template <typename Dummy = Scal>
    std::enable_if_t<SameQ<Real,Dummy>,Real> Min( Int thread_count = 1 ) const
    {
        return min_buffer( a, n, thread_count );
    }

    template <typename Dummy = Scal>
    std::enable_if_t<SameQ<Real,Dummy>,Real> Max( Int thread_count = 1 ) const
    {
        return max_buffer( a, n, thread_count );
    }

    Real MaxNorm( Int thread_count = 1 ) const
    {
        return norm_max( a, n, thread_count );
    }

    Real FrobeniusNorm( Int thread_count = 1 ) const
    {
        return norm_2( a, n, thread_count );
    }

    Real FrobeniusNormSquared( Int thread_count = 1 ) const
    {
        return norm_2_squared( a, n, thread_count );
    }

    friend Real MaxDistance( cref<TENSOR_T> x, cref<TENSOR_T> y )
    {
        cptr<Scal> x_a = x.a;
        cptr<Scal> y_a = y.a;
        
        const Int last = x.Size();
        
        Real max = 0;
        
        for( Int k = 0; k < last; ++k )
        {
            max = Tools::Max( max, Abs( x_a[k] - y_a[k] ) );
        }
        
        return max;
    }

    inline friend Real RelativeMaxError( cref<TENSOR_T> x, cref<TENSOR_T> y )
    {
        return MaxDistance(x,y) / x.MaxNorm();
    }


    template<typename T, typename I, Size_T align>
    TOOLS_FORCE_INLINE mref<TENSOR_T> operator+=( cref<TENSOR_T<T,I,align>> b )
    {
        const Size_T m = Tools::Min( int_cast<Size_T>(n), int_cast<Size_T>(b.Size()) );
        
        combine_buffers<Scalar::Flag::Plus,Scalar::Flag::Plus>(
            Scalar::One<T>, b.data(), Scalar::One<T>, a, m
        );
        
        return *this;
    }

    template<typename T, typename I, Size_T align>
    TOOLS_FORCE_INLINE mref<TENSOR_T> operator-=( cref<TENSOR_T<T,I,align>> b )
    {
        const Size_T m = Tools::Min( int_cast<Size_T>(n), int_cast<Size_T>(b.Size()) );
        
        combine_buffers<Scalar::Flag::Minus,Scalar::Flag::Plus>(
            -Scalar::One<T>, b.data(), Scalar::One<T>, a, m
        );
        
        return *this;
    }

    template<class T>
    TOOLS_FORCE_INLINE mref<TENSOR_T> operator*=( cref<T> alpha )
    {
        scale_buffer( alpha, a, n );
        
        return *this;
    }


    friend void Subtract( cref<TENSOR_T> x, cref<TENSOR_T> y, mref<TENSOR_T> z )
    {
        cptr<Scal> x_a = x.a;
        cptr<Scal> y_a = y.a;
        mptr<Scal> z_a = z.a;
        
        const Int last = x.Size();
        
        for( Int k = 0; k < last; ++ k)
        {
            z_a[k] = x_a[k] - y_a[k];
        }
    }

    friend void Plus( cref<TENSOR_T> x, cref<TENSOR_T> y, mref<TENSOR_T> z )
    {
        cptr<Scal> x_a = x.a;
        cptr<Scal> y_a = y.a;
        mptr<Scal> z_a = z.a;
        
        const Int last = x.Size();
        
        for( Int k = 0; k < last; ++ k)
        {
            z_a[k] = x_a[k] + y_a[k];
        }
    }

    friend void Times( const Scal alpha, cref<TENSOR_T> x, mref<TENSOR_T> y )
    {
        cptr<Scal> x_a = x.a;
        mptr<Scal> y_a = y.a;
        
        const Int last = x.Size();
        
        for( Int k = 0; k < last; ++ k)
        {
            y_a[k] = alpha * x_a[k];
        }
    }

    void Write( std::ostream & s ) const
    {
        s << *this;
    }

    void WriteToFile( const std::filesystem::path & filename ) const
    {
        std::ofstream s ( filename );
        s << *this;
    }

    template<bool verboseQ = true>
    Int Read( std::istream & s )
    {
        Scal number;

        for( Int i = 0; i < n; ++i )
        {
            if( s >> number )
            {
                a[i] = number;
            }
            else
            {
                if constexpr (verboseQ)
                {
                    eprint(ClassName() + "::Read: End of file reached before buffer is filled. Stopped after reading " + ToString(i) + " < " + ToString(n) + " entries.");
                }
                return 1 + i;
            }
        }
        
        if( s >> number )
        {
            if constexpr (verboseQ)
            {
                wprint(ClassName() + "::Read: End of file not reached after buffer is filled.");
            }
            return -2;
        }
        
        return 0;
    }

    template<bool verboseQ = true>
    Int ReadFromFile( const std::filesystem::path & filename )
    {
        std::ifstream s ( filename );
        
        if( s.fail() )
        {
            eprint(ClassName() + "::ReadFromFile failed to load file " + filename.string() + "." );
            
            return -3;
        }
        
        if( s.bad() )
        {
            eprint(ClassName() + "::ReadFromFile: non-recoverable error while loading file " + filename.string() + "." );
            
            return -4;
        }
        
        return Read<verboseQ>(s);
    }


    int WriteToBinaryFile( const std::filesystem::path & filename ) const
    {
        std::ofstream s ( filename );
        
        if( s.fail() )
        {
            eprint(ClassName()+"::WriteToBinaryFile: Failed to open file " + filename.string() + ".");
            
            return 1;
        }
        
        if( !s.write( reinterpret_cast<char*>( a ), ToSize_T(n) * sizeof(Scal) ) )
        {
            TOOLS_DUMP(s.good());
            TOOLS_DUMP(s.fail());
            TOOLS_DUMP(s.eof());
            TOOLS_DUMP(s.bad());
            
            eprint(ClassName()+"::WriteToBinaryFile: Failed to write to file " + filename.string() + ".");
            
            return 2;
        }
        
        return 0;
    }

    int ReadFromBinaryFile( const std::filesystem::path & filename )
    {
        std::ifstream s ( filename );
        
        if( s.fail() || s.bad() )
        {
            eprint(ClassName()+"::ReadFromBinaryFile: Failed to open file " + filename.string() + ".");
            
            return 1;
        }
        
        if( s.eof() )
        {
            eprint(ClassName()+"::ReadFromBinaryFile: File " + filename.string() + " is empty.");
            
            return 2;
        }
        
        s.read( reinterpret_cast<char*>(a), ToSize_T(n) * sizeof(Scal) );
        
        if( s.fail() || s.bad() )
        {
            eprint(ClassName()+"::ReadFromBinaryFile: Failed to read from file " + filename.string() + ".");
            
            return 3;
        }
        
        char c;
        s >> c;
        
        if( !s.eof() )
        {
            eprint(ClassName()+"::ReadFromBinaryFile: Stopped reading from file " + filename.string() + " before end of file.");
            
            return 4;
        }
        
        return 0;
    }



    Size_T AllocatedByteCount() const
    {
        return static_cast<Size_T>(n) * sizeof(Scal);
    }

    Size_T ByteCount() const
    {
        return sizeof(TENSOR_T) + AllocatedByteCount();
    }


    friend bool operator==( cref<TENSOR_T> A, cref<TENSOR_T> B )
    {
        return (A.Size() == B.Size()) && buffers_equalQ(A.data(), B.data(), A.Size());
    }
