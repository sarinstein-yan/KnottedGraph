#pragma once

// To make this work, you should have loaded Tensors/OpenBLAS.hpp or Tensors/Accelerate.hpp.

namespace Tensors
{
    namespace BLAS
    {
        std::string Info()
        {
            std::string s;
            
            s = s + "BLAS::Int           = " + TypeName<Int>              + "\n"
                  + "BLAS::Bool          = " + TypeName<Bool>             + "\n"
                  + "BLAS::ComplexDouble = " + TypeName<ComplexDouble>    + "\n"
                  + "BLAS::ComplexFloat  = " + TypeName<ComplexFloat>;
            
            return s;
        }
        
        // This namespace is to provide wrappers for some BLAS routines.
        
        inline double * to_BLAS( double * x )
        {
            return x;
        }
        
        inline const double * to_BLAS( const double * x )
        {
            return x;
        }
        
        inline float * to_BLAS( float * x )
        {
            return x;
        }
        
        inline const float * to_BLAS( const float * x )
        {
            return x;
        }
        
        inline ComplexDouble * to_BLAS( std::complex<double> * z )
        {
            return reinterpret_cast<ComplexDouble*>(z);
        }
        
        inline const ComplexDouble * to_BLAS( const std::complex<double> * z )
        {
            return reinterpret_cast<const ComplexDouble*>(z);
        }
        
        inline ComplexFloat * to_BLAS( std::complex<float> * z )
        {
            return reinterpret_cast<ComplexFloat*>(z);
        }
        
        inline const ComplexFloat * to_BLAS( const std::complex<float> * z )
        {
            return reinterpret_cast<const ComplexFloat*>(z);
        }
        
        constexpr CBLAS_ORDER to_BLAS( Layout layout )
        {
            if ( layout == Layout::RowMajor )
            {
                return CblasRowMajor;
            }
            else // if ( layout == Layout::RowMajor )
            {
                return CblasColMajor;
            }
        }
        
        constexpr CBLAS_TRANSPOSE to_BLAS( Op op )
        {
            if ( op == Op::Id )
            {
                return CblasNoTrans;
            }
            else if ( op == Op::Trans )
            {
                return CblasTrans;
            }
            else // if ( op == Op::ConjTrans )
            {
                return CblasConjTrans;
            }
        }
        
        constexpr CBLAS_UPLO to_BLAS( UpLo uplo )
        {
            if ( uplo == UpLo::Upper )
            {
                return CblasUpper;
            }
            else // if ( uplo == UpLo::Lower )
            {
                return CblasLower;
            }
        }
        
        constexpr CBLAS_DIAG to_BLAS( Diag diag )
        {
            if ( diag == Diag::NonUnit )
            {
                return CblasNonUnit;
            }
            else // if ( diag == Diagonal::Unit )
            {
                return CblasUnit;
            }
        }
        
        constexpr CBLAS_SIDE to_BLAS( Side side )
        {
            if ( side == Side::Left )
            {
                return CblasLeft;
            }
            else // if ( side == Side::Right )
            {
                return CblasRight;
            }
        }
    }
}

#include "BLAS/scal.hpp"
#include "BLAS/dot.hpp"
#include "BLAS/axpy.hpp"
#include "BLAS/copy.hpp"

#include "BLAS/gemv.hpp"
#include "BLAS/gemm.hpp"

#include "BLAS/trsv.hpp"
#include "BLAS/trsm.hpp"

#include "BLAS/ger.hpp"
#include "BLAS/her.hpp"
#include "BLAS/herk.hpp"
