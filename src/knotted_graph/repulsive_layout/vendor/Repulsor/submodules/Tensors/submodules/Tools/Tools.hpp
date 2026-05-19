#pragma once

#define SCHUMACHER_TOOLS

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    //define something for Windows (32-bit and 64-bit, this part is common)
    #include <io.h>
    #include <stdlib.h>
#else
    #include <unistd.h>
#endif

// Compatibility with non-clang and non-gcc compilers.
#ifndef __has_attribute
  #define __has_attribute(x) 0
#endif

#include <cstddef>
#include <deque>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>
#include <type_traits>
#include <memory>
#include <numeric>
#include <filesystem>
#include <utility>
#include <iomanip>
#include <mutex>
#include <functional>
#include <bit>
#include <concepts>

// In case somebody got the idea to ruin all other people's code.
#ifdef True
    #undef True
#endif

#ifdef False
    #undef False
#endif


// constexpr version of conditional ?-operator
#define COND( condition, case1, case2 ) ([&]{           \
        if constexpr ( condition )                      \
        {                                               \
            return case1;                               \
        }                                               \
        else                                            \
        {                                               \
            return case2;                               \
        }                                               \
    }())

#define TOOLS_STRINGIFY_IMPL(x) #x
#define TOOLS_STRINGIFY(x) TOOLS_STRINGIFY_IMPL(x)

#define TOOLS_TO_STD_STRING(x) std::string(TOOLS_STRINGIFY(x))

#define TOOLS_CONCAT2_IMPL(id1, id2) id1##id2
#define TOOLS_CONCAT2(id1, id2) TOOLS_CONCAT2_IMPL(id1, id2)
    
#define TOOLS_CONCAT3_IMPL(id1, id2, id3) id1##id2##id3
#define TOOLS_CONCAT3(id1, id2, id3) TOOLS_CONCAT3_IMPL(id1, id2, id3)

#define TOOLS_IF #if
#define TOOLS_ELSE #else
#define TOOLS_ENDIF #endif
#define TOOLS_NEWLINE /*
*/

#define TOOLS_DEFINEDQ( S )                                 \
    TOOLS_IF defined((S))          TOOLS_NEWLINE            \
                1                                           \
    TOOLS_ELSE                     TOOLS_NEWLINE            \
                0                                           \
    TOOLS_ENDIF

#ifdef TOOLS_AGGRESSIVE_INLINING

    #if __has_attribute(__always_inline__)
        #define TOOLS_FORCE_INLINE inline __attribute__((__always_inline__))

        #define TOOLS_FORCE_FLATTENING __attribute__((__flatten__))
    #else
        #define TOOLS_FORCE_INLINE inline

        #define TOOLS_FORCE_FLATTENING
    #endif

#else

    #define TOOLS_FORCE_INLINE inline

    #define TOOLS_FORCE_FLATTENING

#endif

// Define loop unrolling depending on the compiler
#ifdef TOOLS_AGGRESSIVE_UNROLLING

    #if defined(__ICC) || defined(__ICL)

        #define TOOLS_LOOP_UNROLL_FULL    _Pragma(TOOLS_STRINGIFY(unroll))
        #define TOOLS_LOOP_UNROLL(n)      _Pragma(TOOLS_STRINGIFY(unroll (n)))

    #elif defined(__clang__)

        #define TOOLS_LOOP_UNROLL_FULL    _Pragma(TOOLS_STRINGIFY(clang loop unroll(enable)))
        #define TOOLS_LOOP_UNROLL(n)      _Pragma(TOOLS_STRINGIFY(clang loop unroll_count(n)))

    #elif defined(__GNUC__) && !defined(__clang__)

        #define TOOLS_LOOP_UNROLL_FULL
        #define TOOLS_LOOP_UNROLL(n)      _Pragma(TOOLS_STRINGIFY(GCC unroll (n)))

    #elif defined(_MSC_BUILD)
        //  #pragma message ("Microsoft Visual C++ (MSVC) detected: Loop unrolling not supported!")
        #define TOOLS_LOOP_UNROLL_FULL
        #define TOOLS_LOOP_UNROLL(n)

    #else
        //  #warning "Unknown compiler: Loop unrolling not supported!"

        #define TOOLS_LOOP_UNROLL_FULL
        #define TOOLS_LOOP_UNROLL(n)
    #endif

#else
    
    #define TOOLS_LOOP_UNROLL_FULL
    #define TOOLS_LOOP_UNROLL(n)

#endif


#if !defined(restrict)
    #if defined(__GNUC__)
        #define restrict __restrict__                     // for gcc
        #ifndef TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
            #define TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT 1
        #endif
    #elif defined(__clang__) && defined(_MSC_VER)         // for clang-cl
        #define restrict __restrict
        #ifndef TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
            #define TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT 1
        #endif
    #elif defined(__clang__) && !defined(_MSC_VER)        // for pure clang
        #define restrict __restrict
        #ifndef TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
            #define TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT 0
        #endif
    #elif !defined(__clang__) && defined(_MSC_VER)        // for pure MSVC
        #define restrict __restrict
        #ifndef TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
            #define TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT 0
        #endif
    #else
        #ifndef TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT
            #define TOOLS_COMPILER_IS_ANAL_ABOUT_RESTRICT 0
        #endif
    #endif
#endif

// Use TOOLS_MAKE_FP_FAST() only in block scope.
#if defined(_MSC_VER)

// see https://learn.microsoft.com/en-us/cpp/preprocessor/float-control?view=msvc-170

#define TOOLS_MAKE_FP_FAST()

#elif defined(__clang__) && !defined(_MSC_VER)        // for pure clang

    #define TOOLS_MAKE_FP_FAST() _Pragma(TOOLS_STRINGIFY(float_control(precise, off)))

#else

    // TODO: Define something better for gcc.

    #define TOOLS_MAKE_FP_FAST()

#endif

// Use TOOLS_MAKE_FP_STRICT() only in block scope.
#if defined(_MSC_VER)

// see https://learn.microsoft.com/en-us/cpp/preprocessor/float-control?view=msvc-170

    #define TOOLS_MAKE_FP_STRICT()     


#elif defined(__clang__) && !defined(_MSC_VER)        // for pure clang

    #define TOOLS_MAKE_FP_STRICT()                              \
        _Pragma(TOOLS_STRINGIFY(float_control(precise, on)))

#else

    // TODO: Define something better for gcc.
    #define TOOLS_MAKE_FP_STRICT()

#endif

namespace Tools
{
    // immutable, unaliased pointer to immutable type
    template<typename T> using cptr = const T * restrict const;

    // immutable, unaliased pointer to mutable type
    template<typename T> using mptr =       T * restrict const;

    // unaliased reference to immutable type
    template<typename T> using cref = const T & restrict;

    // unaliased reference to immutable type
    template<typename T> using mref =       T & restrict;
    
    using Size_T = std::size_t;
    
    constexpr Size_T VarSize = 0;
    
    template<typename T>
    concept IntQ = std::integral<T> || std::integral<typename std::remove_reference<T>::type>;
    
    template<typename T>
    concept NonIntQ = !IntQ<T>;
    
    template<typename T>
    concept IntNotBoolQ = IntQ<T> && !std::is_same_v<T,bool>;
    
#define ASSERT_INT(I) static_assert( IntQ<I>, "Template parameter " #I " must be an integral type." );
    
    template<typename T> 
    concept UnsignedIntQ = IntQ<T> && std::is_unsigned_v<T>;
    
#define ASSERT_UINT(I) static_assert( UnsignedIntQ<I>, "Template parameter " #I " must be a unsigned integral type." );
    
    template<typename T>
    concept SignedIntQ = IntQ<T> && std::is_signed_v<T>;
    
    template<typename T>
    using ToSigned = std::make_signed_t<T>;
    
    template<typename T>
    using ToUnsigned = std::make_unsigned_t<T>;
    
#define ASSERT_SIGNED_INT(I) static_assert( SignedIntQ<I>, "Template parameter " #I " must be a signed integral type." );
    
    template<typename T>
    concept FloatQ = std::floating_point<T>;
    
    template<typename T>
    concept NonPointerQ = !std::is_pointer_v<T> && !std::is_pointer_v<typename std::remove_reference<T>::type>;
    
    template<typename T>
    constexpr bool ArithmeticQ = std::is_arithmetic_v<T>;
    
    template<> constexpr bool ArithmeticQ<std::complex<float>>  = true;
    template<> constexpr bool ArithmeticQ<std::complex<double>> = true;

    template<IntQ I, IntQ J>
    inline bool IntFitsIntoTypeQ( J x )
    {
        return (
            std::cmp_less_equal( x, std::numeric_limits<I>::max() )
            &&
            std::cmp_greater_equal( x, std::numeric_limits<I>::lowest() )
        );
    }
    
    
    template <typename E>
    using Underlying_T = typename std::underlying_type<E>::type;
    
    template <typename E>
    TOOLS_FORCE_INLINE auto constexpr ToUnderlying( const E & e) noexcept
    {
        if constexpr( std::is_enum_v<E> )
        {
            return static_cast<Underlying_T<E>>(e);
        }
        else
        {
            return e;
        }
    }

    
//    // Workaround for static_assert(false,"") within constexpr if.
    template <typename>
    constexpr bool DependentFalse = false;
    

#if ( (!(defined TOOLS_DEACTIVATE_VECTOR_EXTENSIONS)) && (__has_attribute(__ext_vector_type__)) )
    
    constexpr bool vec_enabledQ = true;
    
    template<Size_T N, typename T>
    using vec_T = T __attribute__((__ext_vector_type__(N))) ;

    template<Size_T N, typename T>
    TOOLS_FORCE_INLINE const T * get_ptr( const vec_T<N,T> & vec )
    {
        return reinterpret_cast<const T *>(&vec);
    }
    
    template<Size_T N, typename T>
    TOOLS_FORCE_INLINE T * get_ptr( vec_T<N,T> & vec )
    {
        return reinterpret_cast<T *>(&vec);
    }
    
#else
    constexpr bool vec_enabledQ = false;

    template<Size_T N, typename T>
    using vec_T = std::array<T,N>; //Just a dummy; will not be used, actually.
#endif

#if ( (!(defined TOOLS_DEACTIVATE_MATRIX_EXTENSIONS)) && (__has_attribute(__matrix_type__)) )
    
    constexpr bool mat_enabledQ = true;

    template<Size_T M, Size_T N, typename T>
    using mat_T = T __attribute__((__matrix_type__(M,N))) ;
    
    template<Size_T M, Size_T N, typename T>
    TOOLS_FORCE_INLINE const T * get_ptr( const mat_T<M,N,T> & mat )
    {
        return reinterpret_cast<const T *>(&mat);
    }
    
    template<Size_T M, Size_T N, typename T>
    TOOLS_FORCE_INLINE T * get_ptr( mat_T<M,N,T> & mat )
    {
        return reinterpret_cast<T *>(&mat);
    }
    
#else
    constexpr bool mat_enabledQ = false;

    template<Size_T M, Size_T N, typename T>
    using mat_T = std::array<std::array<T,N>,M>; //Just a dummy; will not be used, actually.
#endif
    
    
    // Dealing with the pain of signed conversions.
    
    template<IntQ Int>
    inline constexpr Size_T ToSize_T( const Int i )
    {
        return static_cast<Size_T>(std::max(Int(0),i));
    }
    
    inline constexpr Size_T ToSize_T( const Size_T i )
    {
        return i;
    }

    /*!@brief Copy-assignment with strong exception guarantee.
     * See this talk by Howard Hinnant: https://www.youtube.com/watch?v=vLinb2fgkHk
     */
    
    template<class T>
    T & strong_assign( T & dest, T src )
    {
        using std::swap;
        swap(dest,src);
        return dest;
    }

} // namespace Tools

#include <complex>
#include "src/ct_string.hpp"
#include "src/ToString.hpp"
#include "src/combine_strings.hpp"

#include "src/FunctionTraits.hpp"
#include "src/OutString.hpp"

#include "src/Print.hpp"
#include "src/Time.hpp"
#include "src/Profiler.hpp"
#include "src/BLAS_Enums.hpp"
#include "src/Integers.hpp"
#include "src/Scalars.hpp"
#include "src/TypeName.hpp"
#include "src/Memory.hpp"


#include "src/BitFiddling.hpp"


#include "src/InString.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include "src/Math/Math.hpp"
#include "src/Math/ProbabilityDistributions.hpp"
#include "src/Math/BisectionSearch.hpp"
#include "src/Math/RegulaFalsi.hpp"
#include "src/Math/Kahan.hpp"
//#include "src/Math/CompensatedSummation.hpp"



//using Tensors::Op;
//using Tensors::AddTo;



#include "src/Scalars/combine_scalars.hpp"

// Common dependencies for combine_scalars_vec and combine_buffers_vec.
#include "src/Scalars/Op_vec.hpp"

// Overloads of Scalar::Op for complex types; uses vec_T.
//#include "src/Scalars/Op_complex.hpp.hpp"

// Overloads of combine_scalars for complex types; uses vec_T.
#include "src/Scalars/combine_scalars_vec.hpp"

// Overloads of combine_scalars for complex types; uses mat_T.
//#include "src/Scalars/combine_scalars_mat.hpp"

#include "src/ParallelDo.hpp"
#include "src/ParallelDoReduce.hpp"
#include "src/ParallelAlgorithms.hpp"
#include "src/JobPointers.hpp"
#include "src/Do.hpp"
#include "src/DoReduce.hpp"

#include "src/BLAS_1.hpp"
#include "src/BLAS_2.hpp"
#include "src/BLAS_4.hpp"

#include "src/AlignedAllocator.hpp"

// TODO: get rid of std::vector here
#include "src/SortedList.hpp"
#include "src/BinarySearch.hpp"

#include "src/CachedObject.hpp"

#include "src/Sort/SortNet.hpp"
//#include "src/Sort/BitonicSort.hpp"
//#include "src/Sort/BitonicMergeSort.hpp"
#include "src/Sort/Sort.hpp"

#include "src/Math/RealCubicSolve.hpp"
#include "src/Math/Det_Bareiss.hpp"
#include "src/Math/ProductAccumulator.hpp"

#include "src/Hashing.hpp"

#include "src/Debugging.hpp"
#include "src/Random.hpp"

