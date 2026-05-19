#pragma once

#include "Base.hpp"

#define TENSORS_ACCELERATE_H

#ifdef __ACCELERATE__

    #ifdef TENSORS_ILP64

        #ifndef ACCELERATE_LAPACK_ILP64
    
            static_assert(false,"Apple Accelerate.h loaded, TENSORS_ILP64 defined, but ACCELERATE_LAPACK_ILP64 undefined. This will result in clashes for the integer types in BLAS and LAPACK.");
        #endif

    #else

        #ifdef ACCELERATE_LAPACK_ILP64

            #define TENSORS_ILP64

        #endif

    #endif
                      
#else

    #ifdef TENSORS_ILP64

        #define ACCELERATE_LAPACK_ILP64

    #endif

    #ifndef TENSORS_USE_ACCELERATE_OLD_LAPACK
        #define ACCELERATE_NEW_LAPACK
    #endif

    #include <Accelerate/Accelerate.h>

#endif

namespace Tensors
{
    constexpr bool AppleAccelerateQ = true;
    constexpr bool OpenBLASQ        = false;
    constexpr bool MKLQ             = false;

    namespace BLAS
    {
        #ifdef ACCELERATE_NEW_LAPACK
            using Int           = __LAPACK_int;
            using Bool          = __LAPACK_bool;
            using ComplexDouble = __LAPACK_double_complex; // std::complex<double>
            using ComplexFloat  = __LAPACK_float_complex;  // std::complex<float>
        #else
            using Int           = __CLPK_integer;
            using Bool          = __CLPK_logical;
            using ComplexDouble = __CLPK_doublecomplex; // a struct with members r, i
            using ComplexFloat  = __CLPK_complex;       // a struct with members r, i
        #endif
    }
    
    namespace LAPACK
    {
        #ifdef ACCELERATE_NEW_LAPACK
            using Int           = __LAPACK_int;
            using Bool          = __LAPACK_bool;
            using ComplexDouble = __LAPACK_double_complex; // std::complex<double>
            using ComplexFloat  = __LAPACK_float_complex;  // std::complex<float>
        #else
            using Int           = __CLPK_integer;
            using Bool          = __CLPK_logical;
            using ComplexDouble = __CLPK_doublecomplex; // a struct with members r, i
            using ComplexFloat  = __CLPK_complex;       // a struct with members r, i
        #endif
    }
}

#include "src/BLAS_Wrappers.hpp"
#include "src/LAPACK_Wrappers.hpp"
