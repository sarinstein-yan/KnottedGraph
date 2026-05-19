#pragma once

#include "Base.hpp"


#ifndef MKL_Complex8

    #define MKL_Complex8 std::complex<float>

#endif

#ifndef MKL_Complex16

    #define MKL_Complex16 std::complex<double>

#endif


    
#ifdef _MKL_H_

    #ifdef TENSORS_ILP64

        #ifndef MKL_ILP64
    
            static_assert(false,"mkl.h loaded, TENSORS_ILP64 defined, but MKL_ILP64 undefined. This will result in clashes for the integer types in BLAS.");
        #endif

    #else

        #ifdef MKL_ILP64

            #define TENSORS_ILP64

        #endif

    #endif
                      
#else

    #ifdef TENSORS_ILP64

        #define MKL_ILP64

    #endif

    #include <mkl.h>

#endif

namespace Tensors
{
    constexpr bool AppleAccelerateQ = false;
    constexpr bool OpenBLASQ        = false;
    constexpr bool MKLQ             = true;
    
    namespace BLAS
    {
        using Int           = MKL_INT;
        using Bool          = bool;
        using ComplexFloat  = MKL_Complex8;
        using ComplexDouble = MKL_Complex16;
    }
    
    namespace LAPACK
    {
        using Int           = MKL_INT;
        using Bool          = bool;
        using ComplexFloat  = MKL_Complex8;
        using ComplexDouble = MKL_Complex16;
    }
}

#include "src/BLAS_Wrappers.hpp"
#include "src/LAPACK_Wrappers.hpp"
