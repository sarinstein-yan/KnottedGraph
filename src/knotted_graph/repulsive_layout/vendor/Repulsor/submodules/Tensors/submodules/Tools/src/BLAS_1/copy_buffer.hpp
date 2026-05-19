#pragma once

namespace Tools
{
    
    template<typename From, typename To>
    struct static_caster
    {
        constexpr To operator()(const From & p)
        {
            return static_cast<To>(p);
        }
    };
    
    
    // This is a bit weird: Typically this fixed-size version should not improve anything because clang and gcc do some heavy optimization for std::copy. In a nutshell, they analyze in every call to std::copy whether n is a compile-time constant -- something that one would not expect a compiler to do for a non-template parameter! However, godbolt tells me that this only happens with at least one restrict qualifier; otherwise the call to std::copy is redirected to std::memmove.
    // We add this fixed-size version anyways so that we do not have to rely on this somewhat unexpected compiler optimization.
    
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        typename S, typename T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void copy_buffer(
        cptr<S> source, mptr<T> target,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        check_sequential<parQ>( "copy_buffer", thread_count );
        
        if constexpr ( N <= VarSize )
        {
            if constexpr ( parQ == Sequential )
            {
                if constexpr ( SameQ<T,S> )
                {
                    std::copy( &source[0], &source[n], &target[0] );
                }
                else
                {
                    std::transform( &source[0], &source[n], &target[0], static_caster<S,T>() );
                }
            }
            else
            {
                if( thread_count <= Int(1) )
                {
                    if constexpr ( SameQ<T,S> )
                    {
                        std::copy( &source[0], &source[n], &target[0] );
                    }
                    else
                    {
                        std::transform( &source[0], &source[n], &target[0], static_caster<S,T>() );
                    }
                }
                else
                {
                    ParallelDo(
                        [=]( const Int thread )
                        {
                            const Int begin = JobPointer(n,thread_count,thread  );
                            const Int end   = JobPointer(n,thread_count,thread+1);

                            if constexpr ( SameQ<T,S> )
                            {
                                std::copy( &source[begin], &source[end], &target[begin] );
                            }
                            else
                            {
                                std::transform( &source[begin], &source[end], &target[begin], static_caster<S,T>() );
                            }
                        },
                        thread_count
                    );
                }
            }
        }
        else
        {
            if constexpr ( SameQ<T,S> )
            {
                std::copy( &source[0], &source[N], &target[0] );
            }
            else
            {
                std::transform( &source[0], &source[N], &target[0], static_caster<S,T>() );
            }
        }
        
    }
    
    
    template <
        Size_T N = VarSize, Parallel_T parQ = Sequential,
        FloatQ S, FloatQ T, IntQ Int = Size_T
    >
    TOOLS_FORCE_INLINE constexpr void copy_buffer_replace_infty(
        cptr<S> source, mptr<T> target,
        const Int n = static_cast<Int>(N),
        const Int thread_count = Int(1)
    )
    {
        check_sequential<parQ>( "copy_buffer_replace_infty", thread_count );
        
        if constexpr ( N <= VarSize )
        {
            if constexpr ( parQ == Sequential )
            {
                TOOLS_MAKE_FP_STRICT();
                
                for( Int i = 0; i < n; ++i )
                {
                    const S x = source[i];

                    if( x == Scalar::Infty<S> )
                    {
                        target[i] = Scalar::Max<T>;
                    }
                    else if( x == -Scalar::Infty<S> )
                    {
                        target[i] = -Scalar::Max<T>;
                    }
                    else
                    {
                        target[i] = static_cast<T>(x);
                    }
                }
            }
            else
            {
                if( thread_count <= Int(1) )
                {
                    copy_buffer_replace_infty<VarSize,Sequential>(source,target,n);
                }
                else
                {
                    ParallelDo(
                        [=]( const Int thread )
                        {
                            TOOLS_MAKE_FP_STRICT();
                            
                            const Int begin = JobPointer(n,thread_count,thread  );
                            const Int end   = JobPointer(n,thread_count,thread+1);

                            copy_buffer_replace_infty<VarSize,Sequential>(
                                &source[begin],&target[begin],end-begin
                            );
                        },
                        thread_count
                    );
                }
            }
        }
        else
        {
            TOOLS_MAKE_FP_STRICT();
            
            for( Int i = 0; i < N; ++i )
            {
                const S x = source[i];

                if( x == Scalar::Infty<S> )
                {
                    target[i] = Scalar::Max<T>;
                }
                else if( x == -Scalar::Infty<S> )
                {
                    target[i] = -Scalar::Max<T>;
                }
                else
                {
                    target[i] = static_cast<T>(x);
                }
            }
        }
    }
    
} // namespace Tools
