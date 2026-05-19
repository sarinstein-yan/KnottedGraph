#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        
        template<Layout layout, UpLo uplo, typename Scal>
        class SelfAdjointEigensolver final
        {
            
        public:
            
            SelfAdjointEigensolver() = default;
            
        private:
            
            // Caution: Older versions of Apple Accelerate expect this to be a non-const pointer!
            char flag = to_LAPACK(
                layout == Layout::ColMajor ? uplo : Transpose(uplo)
            );
            
            Int n;
            
            Int ldA;
            
            Int lwork;
            
            Tensor1<Scalar::Real<Scal>,Int> eigs;
            
            Tensor2<Scal,Int> A;
            
            Tensor1<Scal,Int> work;
            
            Tensor1<Scalar::Real<Scal>,Int> rwork;
            
        public:
            
            template<IntQ I0, IntQ I1>
            Int Eigenvalues( const I0 n_, mptr<Scal> A_, const I1 ldA_, mptr<Scalar::Real<Scal>> eigs_ )
            {
                n    = int_cast<Int>(n_);
                ldA  = int_cast<Int>(ldA_);
                
                Int info = Prepare( 'N', A_ );
                
                info = heev( 'N');
                
                eigs.Write(eigs_);
                
                return info;
            }
            
            template< typename S, IntQ I0, IntQ I1, IntQ I2>
            Int Eigensystem( const I0 n_, mptr<S> A_, const I1 ldA_, mptr<Scalar::Real<S>> eigs_, mptr<S> Q_, const I2 ldQ )
            {
                // Returns Q and eigs such that ConjugateTranspose(Q) * A * Q == Diagona(eigs);
                
                n    = int_cast<Int>(n_);
                ldA  = int_cast<Int>(ldA_);
                
                Int info = Prepare( 'V', A_ );
                
                info = heev( 'V' );
                
                eigs.Write(eigs_);
                
                if constexpr ( layout == Layout::RowMajor )
                {
                    A.template Write<Op::ConjTrans>( Q_, int_cast<Int>(ldQ) );
                }
                else
                {
                    A.template Write<Op::Id>( Q_, int_cast<Int>(ldQ) );
                }
                
                return info;
            }
            
            
        private:
            
            
            template< typename S>
            TOOLS_FORCE_INLINE Int Prepare( char job, mptr<S> A_ )
            {
                assert_positive(n);
                assert_positive(ldA);
                
                work.template RequireSize<false>( 1 );
                work[0] = 0;
                
                if constexpr ( Scalar::ComplexQ<Scal> )
                {
                    rwork.template RequireSize<false>( 3 * n - 2 );
                }
                
                Int info;
                
                lwork = -1;
                
                // Request the required work space from LAPACK
                eigs.template RequireSize<false>( n );
                
                A.template RequireSize<false>( n, n );
                
                A.Read( A_, ldA );
                
                info = heev( job );
                
                lwork = static_cast<int>(Re(work[0]));
                                
                work.template RequireSize<false>( lwork );
                
                return info;
            }
            
            TOOLS_FORCE_INLINE Int heev( char job  )
            {
                Int info = 0;
                
                auto * A_     = to_LAPACK(A.data());
                auto * eigs_  = to_LAPACK(eigs.data());
                auto * work_  = to_LAPACK(work.data());
                auto * rwork_ = to_LAPACK(rwork.data());
                                
                if constexpr ( SameQ<Scal,double> )
                {
                    #ifdef LAPACK_dsyev
                        LAPACK_dsyev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #else
                        dsyev_      ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #endif
                }
                else if constexpr ( SameQ<Scal,float> )
                {
                    #ifdef LAPACK_ssyev
                        LAPACK_ssyev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #else
                        ssyev_      ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #endif
                }
                else if constexpr ( SameQ<Scal,std::complex<double>> )
                {
                    #ifdef LAPACK_zheev
                        LAPACK_zheev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #else
                        zheev_      ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #endif
                }
                else if constexpr ( SameQ<Scal,std::complex<float>> )
                {
                    #ifdef LAPACK_cheev
                        LAPACK_cheev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #else
                        cheev       ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #endif
                }
                else
                {
                    eprint("heev not defined for scalar type " + TypeName<Scal> );
                }

                return info;
            }
        };
        
        
        
    } // namespace LAPACK
    
} // namespace Tensors

