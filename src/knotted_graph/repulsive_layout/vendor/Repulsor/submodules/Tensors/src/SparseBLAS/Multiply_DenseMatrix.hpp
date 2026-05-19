//###########################################################
//####                  General matrices                      
//###########################################################

public:

    // Calling this NRHS <= 0 will use generic implementation.
    // Calling with NRHS > 0  will uses dimension-dependent implementation
    template<Int NRHS = VarSize, bool base = 0, typename alpha_T, typename X_T, typename beta_T, typename Y_T>
    void Multiply_DenseMatrix
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const alpha_T alpha, cptr<X_T> X, const Int ldX,
        const beta_T  beta,  mptr<Y_T> Y, const Int ldY,
        const Int nrhs,
        cref<JobPointers<Int>> job_ptr
    )
    {
        if( (NRHS > VarSize) && ( nrhs != NRHS ) )
        {
            eprint( ClassName() + "Multiply_DenseMatrix: nrhs != NRHS. Doing nothing." );
            TOOLS_DUMP(NRHS);
            TOOLS_DUMP(nrhs);
            
            return;
        }
        
        if( nrhs == Scalar::One<Int> )
        {
            if( (ldX == Scalar::One<Int>) && (ldY == Scalar::One<Int>) )
            {
                SpMV<base>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
            }
            else
            {
                SpMM<1,base>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
            }

            return;
        }
        else
        {
            SpMM<NRHS,base>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
        }
    }
