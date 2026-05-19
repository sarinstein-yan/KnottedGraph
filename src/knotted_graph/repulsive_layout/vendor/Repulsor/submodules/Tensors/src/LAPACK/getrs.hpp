#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout_A, Op op_A,
            Layout layout_Y = layout_A,
            Layout layout_X = layout_Y,
            typename Scal, IntQ I0, IntQ I1, IntQ I2, IntQ I3, IntQ I4
        >
        TOOLS_FORCE_INLINE Int getrs(
            const I0 n_, const I1 nrhs_,
            Scal * A_, const I2 ldA_,
            Int * perm,
            Scal * Y_, const I3 ldY_,
            Scal * X_, const I4 ldX_
        )
        {
            Int n    = int_cast<Int>(n_);
            Int nrhs = int_cast<Int>(nrhs_);
            Int ldA  = int_cast<Int>(ldA_);
            Int ldB  = int_cast<Int>(n);
            Int info = 0;
            
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(ldB);
            
            // We always allocated at copy twice, but it is really, really cheap compared to the matrix solve.
            Tensor2<Scal,Size_T> B_buffer (nrhs, n);
            
            if( layout_Y == Layout::RowMajor  )
            {
                B_buffer.template Read<Op::Trans>(Y_, ldY_);
            }
            else
            {
                B_buffer.template Read<Op::Id>(Y_, ldY_);
            }
            
            auto * A = to_LAPACK(A_);
            auto * B = to_LAPACK(B_buffer.data());
            
            char transA = (layout_A == Layout::ColMajor)
                        ? to_LAPACK(op_A)
                        : to_LAPACK(Transpose(op_A));
            
            
//            TOOLS_DUMP(layout);
//            TOOLS_DUMP(op_A);
//            TOOLS_DUMP(n);
//            TOOLS_DUMP(nrhs);
//            TOOLS_DUMP(ldA);
//            TOOLS_DUMP(ldB);
//            TOOLS_DUMP(transA);
            
            if constexpr ( SameQ<Scal,double> )
            {
#if defined LAPACK_dgetrs
                LAPACK_dgetrs( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#else
                dgetrs_      ( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
#if defined LAPACK_sgetrs
                LAPACK_sgetrs( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#else
                sgetrs_      ( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
#if defined LAPACK_zgetrs
                LAPACK_zgetrs( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#else
                zgetrs_      ( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
#if defined LAPACK_cgetrs
                LAPACK_cgetrs( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#else
                cgetrs_      ( &transA, &n, &nrhs, A, &ldA, perm, B, &ldB, &info );
#endif
            }
            else
            {
                eprint("getrs not defined for scalar type " + TypeName<Scal> );
            }
            
            if( layout_X == Layout::RowMajor  )
            {
                B_buffer.template Write<Op::Trans>(X_, ldX_);
            }
            else
            {
                B_buffer.template Write<Op::Id>(X_, ldX_);
            }
            
            return info;
        }
        
        template<
            Layout layout_A, Op op_A, Layout layout_B = layout_A,
            typename Scal, typename I0, typename I1, typename I2, typename I3
        >
        TOOLS_FORCE_INLINE Int getrs(
            const I0 n_, const I1 nrhs_,
            Scal * A_, const I2 ldA_,
            Int * perm,
            Scal * B_, const I3 ldB_
        )
        {
            return getrs<layout_A,op_A,layout_B>(
                n_, nrhs_, A_, ldA_, perm, B_, ldB_, B_, ldB_
            );
        }
        
    } // namespace LAPACK

    
} // namespace Tensors




