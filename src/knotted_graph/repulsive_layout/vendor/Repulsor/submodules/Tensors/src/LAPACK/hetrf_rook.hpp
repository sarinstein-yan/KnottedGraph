#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout, UpLo uplo,
            typename Scal, IntQ I1, IntQ I2, IntQ I3
        >
        TOOLS_FORCE_INLINE Int hetrf_rook(
            const I1 n_, Scal * A_, const I2 ldA_,
            Scal * work_, const I3 lwork_,
            Int * perm
        )
        {
            Int n       = int_cast<Int>(n_);
            Int ldA     = int_cast<Int>(ldA_);
            Int lwork   = int_cast<Int>(lwork_);
            Int info    = 0;
            
            auto * A    = to_LAPACK(A_);
            auto * work = to_LAPACK(work_);
            
            // Caution: Older versions of Apple Accelerate expect this to be a non-const pointer!
            char flag = to_LAPACK(
                layout == Layout::ColMajor ? uplo : Transpose(uplo)
            );
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dsytrf_rook
                    LAPACK_dsytrf_rook( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    dsytrf_rook_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_ssytrf_rook
                    LAPACK_ssytrf_rook( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    ssytrf_rook_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zhetrf_rook
                    LAPACK_zhetrf_rook( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    zhetrf_rook_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_chetrf_rook
                    LAPACK_chetrf_rook( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    chetrf_rook_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else
            {
                eprint("hetrf_rook not defined for scalar type " + TypeName<Scal> );
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors




