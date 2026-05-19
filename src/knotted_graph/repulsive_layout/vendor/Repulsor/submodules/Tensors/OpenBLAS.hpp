#pragma once

#include "Base.hpp"

#define TENSORS_OPENBLAS_H

#ifdef CBLAS_H

    #ifdef TENSORS_ILP64

        #ifndef OPENBLAS_USE64BITINT
    
            static_assert(false,"cblas.h loaded, TENSORS_ILP64 defined, but OPENBLAS_USE64BITINT undefined. This will result in clashes for the integer types in BLAS.");
        #endif

    #else

        #ifdef OPENBLAS_USE64BITINT

            #define TENSORS_ILP64

        #endif

    #endif
                      
#else

    #ifdef TENSORS_ILP64

        #define OPENBLAS_USE64BITINT

    #endif

    #include <cblas.h>

#endif

#ifdef LAPACK_H

    #ifdef TENSORS_ILP64

        #ifndef LAPACK_ILP64

            static_assert(false,"cblas.h loaded, TENSORS_ILP64 defined, but LAPACK_ILP64 undefined. This will result in clashes for the integer types in BLAS and LAPACK.");
        #endif

    #else

        #ifdef LAPACK_ILP64

            #define TENSORS_ILP64

        #endif

    #endif
                      
#else

    #ifdef TENSORS_ILP64

        #define LAPACK_ILP64

    #endif

    #include <lapack.h>

#endif

namespace Tensors
{
    constexpr bool AppleAccelerateQ = false;
    constexpr bool OpenBLASQ        = true;
    constexpr bool MKLQ             = false;
    
    namespace BLAS
    {
        using Int           = blasint;
        using Bool          = bool;
        using ComplexFloat  = lapack_complex_float;
        using ComplexDouble = lapack_complex_double;
    }
    
    namespace LAPACK
    {
        using Int           = lapack_int;
        using Bool          = bool;
        using ComplexFloat  = lapack_complex_float;
        using ComplexDouble = lapack_complex_double;
    }
}

#include "src/BLAS_Wrappers.hpp"
#include "src/LAPACK_Wrappers.hpp"
