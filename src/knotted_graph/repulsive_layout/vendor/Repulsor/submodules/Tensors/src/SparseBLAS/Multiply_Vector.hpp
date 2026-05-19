public:

    template<bool base = 0, typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const R_out alpha, cptr<T_in>  X,
        const S_out beta,  mptr<T_out> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        SpMV<base>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<bool base = 0, typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const R_out alpha, cptr<T_in>  X, const Int incX,
        const S_out beta,  mptr<T_out> Y, const Int incY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        if( (incX == 1) && (incY == 1) )
        {
            SpMV<base>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
        }
        else
        {
            SpMM<1,base>(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
        }
    }


    template<bool base = 0, typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector_Transposed
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const R_out alpha, cptr<T_in>  X,
        const S_out beta,  mptr<T_out> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        SpMV_Transposed<base>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<bool base = 0, typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector_Transposed
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const R_out alpha, cptr<T_in>  X, const Int incX,
        const S_out beta,  mptr<T_out> Y, const Int incY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        if( (incX == 1) && (incY == 1) )
        {
            SpMV_Transpose<base>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
        }
        else
        {
            SpMM_Transposed<1,base>(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
        }
    }
