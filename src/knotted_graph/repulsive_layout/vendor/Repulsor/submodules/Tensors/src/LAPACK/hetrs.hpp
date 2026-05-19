#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout, UpLo uplo,
            typename Scal, IntQ I0, IntQ I1, IntQ I2, IntQ I3
        >
        TOOLS_FORCE_INLINE Int hetrs(
            const I0 n_, const I1 nrhs_,
            Scal * A_, const I2 ldA_,
            Int * perm,
            Scal * B_, const I3 ldB_
        )
        {
            Int n       = int_cast<Int>(n_);
            Int nrhs    = int_cast<Int>(nrhs_);
            Int ldA     = int_cast<Int>(ldA_);
            Int ldB     = int_cast<Int>(ldB_);
            Int info    = 0;
            
            auto * A    = to_LAPACK(A_);
            auto * B    = to_LAPACK(B_);
            
            // Caution: Older versions of Apple Accelerate expect this to be a non-const pointer!
            char flag = to_LAPACK(
                layout == Layout::ColMajor ? uplo : Transpose(uplo)
            );
            
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(ldB);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dsytrs
                    LAPACK_dsytrs( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #else
                    dsytrs_      ( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_ssytrs
                    LAPACK_ssytrs( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #else
                    ssytrs_      ( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zhetrs
                    LAPACK_zhetrs( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #else
                    zhetrs_      ( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_chetrs
                    LAPACK_chetrs( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #else
                    chetrs_      ( &flag, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
                #endif
            }
            else
            {
                eprint("hetrs not defined for scalar type " + TypeName<Scal> );
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors



