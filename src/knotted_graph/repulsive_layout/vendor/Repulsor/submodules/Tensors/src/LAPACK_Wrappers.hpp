#pragma once

// To make this work, you should have loaded Tensors/OpenBLAS.hpp or Tensors/Accelerate.hpp.

namespace Tensors
{
    namespace LAPACK
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
        
        inline double * to_LAPACK( double * x )
        {
            return x;
        }
        
        inline const double * to_LAPACK( const double * x )
        {
            return x;
        }
        
        inline float * to_LAPACK( float * x )
        {
            return x;
        }
        
        inline const float * to_LAPACK( const float * x )
        {
            return x;
        }
        
        inline ComplexDouble * to_LAPACK( std::complex<double> * z )
        {
            return reinterpret_cast<ComplexDouble*>(z);
        }
        
        inline const ComplexDouble * to_LAPACK( const std::complex<double> * z )
        {
            return reinterpret_cast<const ComplexDouble*>(z);
        }
        
        inline ComplexFloat * to_LAPACK( std::complex<float> * z )
        {
            return reinterpret_cast<ComplexFloat*>(z);
        }
        
        inline const ComplexFloat * to_LAPACK( const std::complex<float> * z )
        {
            return reinterpret_cast<const ComplexFloat*>(z);
        }
        
        // This namespace is to provide wrappers for some LAPACK routines.
        
        constexpr int to_LAPACK( Layout layout )
        {
            if ( layout == Layout::RowMajor )
            {
                return 101; // LAPACK_ROW_MAJOR
            }
            else // if ( layout == Layout::RowMajor )
            {
                return 102; // LAPACK_COL_MAJOR
            }
        }
        
        constexpr char to_LAPACK( Op op )
        {
            if ( op == Op::Id )
            {
                return 'N';
            }
            else if ( op == Op::Trans )
            {
                return 'T';
            }
            else // if ( op == Op::ConjTranspose )
            {
                return 'C';
            }
        }
        
        constexpr char to_LAPACK( UpLo uplo )
        {
            if ( uplo == UpLo::Upper )
            {
                return 'U';
            }
            else // if ( uplo == UpLo::Lower )
            {
                return 'L';
            }
        }
    }
}

#include "LAPACK/potrf.hpp"
#include "LAPACK/potrs.hpp"

#include "LAPACK/getrf.hpp"
#include "LAPACK/getrs.hpp"

#include "LAPACK/hetrf.hpp"
#include "LAPACK/hetrs.hpp"

#include "LAPACK/hetrf_rk.hpp"

#include "LAPACK/hetrf_rook.hpp"
#include "LAPACK/hetrs_rook.hpp"

#include "LAPACK/SelfAdjointEigensolver.hpp"

