#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout, UpLo uplo,
            bool verboseQ = true,
            typename Scal, IntQ I0, IntQ I1
        >
        TOOLS_FORCE_INLINE Int potrf( const I0 n_, Scal * A_, const I1 ldA_ )
        {
            Int n    = int_cast<Int>(n_);
            Int ldA  = int_cast<Int>(ldA_);
            Int info = 0;
            
            auto * A = to_LAPACK(A_);
            
            // Caution: Older versions of Apple Accelerate expect this to be a non-const pointer!
            char flag = to_LAPACK(
                layout == Layout::ColMajor
                ? uplo
                : Transpose(uplo)
            );
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dpotrf
                    LAPACK_dpotrf( &flag, &n, A, &ldA, &info );
                #else
                    dpotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_spotrf
                    LAPACK_spotrf( &flag, &n, A, &ldA, &info );
                #else
                    spotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zpotrf
                    LAPACK_zpotrf( &flag, &n, A, &ldA, &info );
                #else
                    zpotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_cpotrf
                    LAPACK_cpotrf( &flag, &n, A, &ldA, &info );
                #else
                    cpotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else
            {
                eprint("potrf not defined for scalar type " + TypeName<Scal> );
            }
            
            if( info != Int(0) )
            {
                std::string tag = std::string("BLAS::potrf")
                    + "<" + ToString(layout)
                    + "," + ToString(uplo)
                    + "," + TypeName<Scal>
                    + ">(" + ToString(n) + ")";
                
                if( info < Int(0) )
                {
                    eprint( tag + ": input " + ToString(-info) + " is invalid." );
                }
                else
                {
                    if constexpr( verboseQ )
                    {
                        eprint( tag + ": The leading minor of order " + ToString(info) + " is not positive-definite." );
                        if( info <= Int(16) )
                        {
                            logvalprint("leading minor", OutString::FromMatrix(A_, info, info) );
                        }
                    }
                }
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors
