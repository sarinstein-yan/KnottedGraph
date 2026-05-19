#pragma once

namespace Tools
{
    #if !defined(prefetch)
        #if defined(__GNUC__) || defined(__clang__)
            #define prefetch __builtin_prefetch
        #else
            #define prefetch( a, b, c )
        #endif
#endif // !defined(prefetch)
    
    // length of cache line measured in bytes
    constexpr Size_T CacheLineWidth = 64;
//    //TODO: Replace the hard-coded constant by __cpp_lib_hardware_interference_size as soon as this C++17 feature is supported by all compilers.
    
//    constexpr Size_T CacheLineWidth = std::hardware_destructive_interference_size;
    
    constexpr Size_T DefaultAlignment = std::max( sizeof(void*), std::size_t(16) );

    constexpr Size_T ObjectAlignment = std::max( CacheLineWidth, DefaultAlignment );

    constexpr Size_T PrefetchStride = 4 * CacheLineWidth;
    
    
    template<IntQ Int>
    TOOLS_FORCE_INLINE constexpr Int CeilDivide( const Int n, const Int b )
    {
        return ( (n + b - Int(1) ) / b );
    }

    template<IntQ Int>
    TOOLS_FORCE_INLINE constexpr Int RoundUpTo( const Int n, const Int b )
    {
        return CeilDivide(n,b) * b;
    }
    
    
    template<IntQ Int>
    TOOLS_FORCE_INLINE constexpr Int FloorDivide( const Int n, const Int b )
    {
        return n / b;
    }
    
    template<IntQ Int>
    TOOLS_FORCE_INLINE constexpr Int RoundDownTo( const Int n, const Int b )
    {
        return FloorDivide(n,b) * b;
    }
    
    // Computes k-th job pointer for job_count equally sized jobs distributed on thread_count threads.
    template<IntQ Int, IntQ Int1, IntQ Int2>
    TOOLS_FORCE_INLINE Int JobPointer( const Int job_count, const Int1 thread_count_, const Int2 thread_ )
    {
        const Int thread_count = static_cast<Int>(thread_count_);
        const Int thread       = static_cast<Int>(thread_);
        const Int chunk_size   = job_count / thread_count;
        const Int remainder    = job_count % thread_count;
        
        return chunk_size * thread + (remainder * thread) / thread_count;
    }
    
    inline bool is_aligned( const void * const pointer, const Size_T byte_count = DefaultAlignment )
    {
        return reinterpret_cast<std::uintptr_t>(pointer) % byte_count == 0;
    }
    
    inline Size_T get_alignment( const void * const pointer )
    {
        std::uintptr_t ptr = reinterpret_cast<std::uintptr_t>(pointer);

        return (Size_T(1) << std::countr_zero(ptr));
    }
    
    [[nodiscard]] TOOLS_FORCE_INLINE void * aligned_malloc( const Size_T size, const Size_T alignment )
    {
        const Size_T padded_size = RoundUpTo(size, alignment);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        void * ptr_ = _aligned_malloc(padded_size, alignment);
#else
        void * ptr_ = std::aligned_alloc(alignment, padded_size);
#endif

        if( ptr_ == nullptr )
        {
            eprint("aligned_malloc: failed to allocate memory.");
            TOOLS_DUMP(size);
            TOOLS_DUMP(alignment);
            TOOLS_DUMP(padded_size);
            return nullptr;
        }
        else
        {
            return ptr_;
        }
    }
    
    TOOLS_FORCE_INLINE void aligned_free( void * ptr_ )
    {
        #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
            _aligned_free( ptr_ );
        #else
            std::free( ptr_ );
        #endif
    }



    template <typename T>
    TOOLS_FORCE_INLINE int safe_free( T * & ptr_ )
    {
        int wasallocated = (ptr_ != nullptr);
        if( wasallocated )
        {
            aligned_free(ptr_);
            ptr_ = nullptr;

        }
        return !wasallocated;
    }

#if TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
    // overload function for restrict qualifier
    template <typename T>
    TOOLS_FORCE_INLINE int safe_free( T * restrict & ptr_ )
    {
        int wasallocated = (ptr_ != nullptr);
        if( wasallocated )
        {
            aligned_free(ptr_);
            ptr_ = nullptr;
        }
        return !wasallocated;
    }
#endif


    template <typename T>
    TOOLS_FORCE_INLINE int safe_alloc(
        T * & ptr_, const Size_T n, const Size_T aligment = DefaultAlignment
    )
    {
        int wasallocated = (ptr_ != nullptr);

        if( wasallocated != 0 )
        {
#ifdef SAFE_ALLOCATE_WARNINGS
            wprint("safe_alloc: Pointer was not nullptr. Calling safe_free to prevent memory leak.");
#endif
            safe_free(ptr_);
        }

        std::size_t max_n = std::numeric_limits<std::size_t>::max() / sizeof(T);
        
        if( n > max_n )
        {
            eprint(std::string("safe_alloc<") + TypeName<T> + ": size n = " + ToString(n) + " was too big to fit into memory.");
            return 2;
        }
        
        ptr_ = static_cast<T *>( aligned_malloc( n * sizeof(T), aligment ) );

        return wasallocated;
    }

#if TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
    // overload function for restrict qualifier
    template <typename T>
    TOOLS_FORCE_INLINE int safe_alloc(
        T * restrict & ptr_, const Size_T n, const Size_T aligment = DefaultAlignment
    )
    {
        int wasallocated = (ptr_ != nullptr);

        if( wasallocated != 0 )
        {
        #ifdef SAFE_ALLOCATE_WARNINGS
            wprint("safe_alloc: Pointer was not nullptr. Calling safe_free to prevent memory leak.");
        #endif
            safe_free(ptr_);
        }

        ptr_ = static_cast<T *>( aligned_malloc( n * sizeof(T), aligment ) );

        return wasallocated;
    }
#endif
    
    
    template<int readwrite, int locality, typename T>
    TOOLS_FORCE_INLINE void prefetch_buffer( cptr<T> begin, const Size_T n )
    {
        const Size_T prefetch_size = n * sizeof(T);
        
        const unsigned char * ptr_ = reinterpret_cast<const unsigned char*>(begin);
    
        for( Size_T offset = 0; offset < prefetch_size; offset += PrefetchStride )
        {
            prefetch( &ptr_[offset], readwrite, locality );
        }
    }
    
    template<Size_T N, int readwrite, int locality, typename T>
    TOOLS_FORCE_INLINE void prefetch_buffer( cptr<T> begin )
    {
        constexpr Size_T prefetch_size = N * sizeof(T);
        
        const unsigned char * ptr_ = reinterpret_cast<const unsigned char*>(begin);
    
        TOOLS_LOOP_UNROLL_FULL
        for( Size_T offset = 0; offset < prefetch_size; offset += PrefetchStride )
        {
            prefetch( &ptr_[offset], readwrite, locality );
        }
    }
    
} // namespace Tools
