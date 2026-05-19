#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout, UpLo uplo, 
            typename Scal, IntQ I0, IntQ I1, IntQ I2, IntQ I3
        >
        TOOLS_FORCE_INLINE Int potrs( I0 n_, I1 nrhs_, Scal * A_, I2 ldA_, Scal * B_, I3 ldB_ )
        {
            Int n    = int_cast<Int>(n_);
            Int nrhs = int_cast<Int>(nrhs_);
            Int ldA  = int_cast<Int>(ldA_);
            Int ldB  = int_cast<Int>(ldB_);
            Int info = 0;
            
            auto * A = to_LAPACK(A_);
            auto * B = to_LAPACK(B_);
            
            // Caution: Older versions of Apple Accelerate expect this to be a non-const pointer!
            char flag = to_LAPACK(
                layout == Layout::ColMajor ? uplo : Transpose(uplo)
            );
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dpotrs
                    LAPACK_dpotrs( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #else
                    dpotrs_      ( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_spotrs
                    LAPACK_spotrs( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #else
                    spotrs_      ( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zpotrs
                    LAPACK_zpotrs( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #else
                    zpotrs_      ( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_cpotrs
                    LAPACK_cpotrs( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #else
                    cpotrs_      ( &flag, &n, &nrhs, A, &ldA, B, &ldB, &info );
                #endif
            }
            else
            {
                eprint("potrs not defined for scalar type " + TypeName<Scal> );
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors

