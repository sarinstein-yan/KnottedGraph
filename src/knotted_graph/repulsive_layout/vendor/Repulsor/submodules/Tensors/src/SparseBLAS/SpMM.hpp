public:

    template<
        Size_T NRHS = VarSize, bool base,
        typename alpha_T, typename X_T, typename beta_T, typename Y_T
    >
    void SpMM(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<X_T> X, const Int ldX,
        cref<beta_T>  beta,  mptr<Y_T> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr,
        const Int nrhs = NRHS
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.
        
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if( alpha == alpha_T(0) )
        {
            modify_matrix_auto<VarSize,NRHS,Parallel>(
                beta, Y, ldY, m, nrhs, job_ptr.ThreadCount()
            );
            return;
        }
        
        auto job = [=,this]<F_T a_flag,F_T alpha_flag,F_T beta_flag>()
        {
            this->SpMM_impl<a_flag,alpha_flag,beta_flag,NRHS,base>(
                rp, ci, a, m, n,
                alpha, X, ldX,
                beta,  Y, ldY,
                job_ptr, nrhs
            );
        };
        
        if( a != nullptr )
        {
            constexpr F_T a_flag = F_T::Generic;
            
            if( alpha == alpha_T(1) )
            {
                constexpr F_T alpha_flag = F_T::Plus;
                
                if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
            else
            {
                constexpr F_T alpha_flag = F_T::Generic;
                
                // general alpha
                if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
        }
        else // a == nullptr
        {
            constexpr F_T a_flag = F_T::Plus;
            
            if( alpha == alpha_T(1) )
            {
                constexpr F_T alpha_flag = F_T::Plus;
                
                if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
            else
            {
                // general alpha
                
                constexpr F_T alpha_flag = F_T::Generic;
                
                if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
        }
    }

private:
  

    template<
        F_T a_flag, F_T alpha_flag, F_T beta_flag,
        Size_T NRHS = VarSize, bool base = 0,
        typename alpha_T, typename X_T, typename beta_T, typename Y_T
    >
    void SpMM_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<X_T> X, const Int ldX,
        cref<beta_T>  beta,  mptr<Y_T> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr,
        const Int nrhs = NRHS
    )
    {
        using namespace Scalar;
        
        // At least with fast-math flag the generic code is exactly as fast as this one.
        // But this has vectorization guarantees, even without fast-math.
        if constexpr(
            (NRHS > VarSize) && ( a_flag != F_T::Zero )
            &&
            VectorizableQ<Y_T>
            &&
            std::is_same_v<Scal,Y_T>
            &&
            std::is_same_v<X_T,Y_T>
            &&
            std::is_same_v<alpha_T,X_T>
            &&
            std::is_same_v<beta_T,Y_T>
        )
        {
            return SpMM_vec<a_flag,alpha_flag,beta_flag,NRHS,base>(
                rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr
           );
        }

        std::string tag = std::string(ClassName()+"::SpMM_impl")
            + "<" + ToString(a_flag)
            + "," + ToString(alpha_flag)
            + "," + ToString(beta_flag)
            + "," + (NRHS <= 0 ? std::string("VarSize") : ToString(NRHS) )
            + "," + ToString(base)
            + "," + TypeName<alpha_T>
            + "," + TypeName<X_T>
            + "," + TypeName<beta_T>
            + "," + TypeName<Y_T>
            +">(" + ToString(ldX)
            + "," + ToString(ldY)
            + "," + ToString(nrhs)
            + ")";
        
        TOOLS_PTIMER(timer,tag);
        
        // TODO: Add better check for pointer overlap.
        
        if constexpr ( std::is_same_v<X_T,Y_T> )
        {
            if( X == Y )
            {
                eprint( tag + ": Input and output pointer coincide. This is not safe. Aborting.");
                return;
            }
        }
        
        
        // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == One
        //  - alpha_flag == One
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic

        // Treats sparse matrix as a binary matrix if a_flag != F_T::Generic.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.

        if constexpr ( a_flag == F_T::Generic )
        {
            if ( a == nullptr )
            {
                eprint( tag + ": a_flag == F_T::Generic, but the pointer a is a nullptr. Aborting.");
                return;
            }
        }
        
        // We have to do z += a * x quite often.
        // a could be complex or real.
        // x could be complex or real.
        // a*x is real only if a_T and X_T are real.
        //
        // Which precision to use?
        // We have the following options:
        // -- use min( Prec<a_T>, Prec<X_T> );
        // -- use max( Prec<a_T>, Prec<X_T> );
        // -- use Prec<a_T>;
        // -- use Prec<X_T>;
        
        // With many right-hand sides, casting x will be more expensive
        // than casting a.
        
        // Make precisions of a and X compatible.
        // Using precision of X to minimize casting during runtime.
        using a_T = std::conditional_t<
            Scalar::ComplexQ<Scal>,
            Scalar::Complex<X_T>,
            Scalar::Real<X_T>
        >;
        
        // Define z_T so that it can hold a * x.
        using z_T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<X_T>,
            typename Scalar::Complex<X_T>,
            typename Scalar::Real<X_T>
        >;

        // Check whether z = a * x will work.
//            StaticParameterCheck<Scalar::Real<z_T>,z_T,a_T,X_T>();

        // Not as often we have to do y = alpha * z + beta * y;
        // Nonetheless, we should cast alpha and beta to specific
        // types to make this efficient.
        //
        // The current implementation of combine_scalars computes this
        // by y = ( alpha * z) + (beta * y);
        //
        
        using alpha_T_ = std::conditional_t<
            Scalar::ComplexQ<alpha_T>,
            Scalar::Complex<z_T>,
            Scalar::Real<z_T>
        >;
        
        using beta_T_ = std::conditional_t<
            Scalar::ComplexQ<beta_T>,
            Scalar::Complex<Y_T>,
            Scalar::Real<Y_T>
        >;
        
        const alpha_T_ alpha_ = static_cast<alpha_T_>(alpha);
        const beta_T_  beta_  = static_cast<beta_T_ >(beta );
        
        
        // Check whether y = alpha_ * z + beta_ * y will work.
//            StaticParameterCheck<alpha_T_,z_T,beta_T_,Y_T>();
        
        constexpr bool prefetchQ = true;
        
        ParallelDo(
            [&]( const Int thread )
            {
                std::conditional_t<
                    NRHS==VarSize,
                    Tensor1<z_T,Int>,
                    Tiny::Vector<NRHS,z_T,Int>
                > z;
                
                if constexpr ( NRHS==VarSize )
                {
                    z = Tensor1<z_T,Int>( nrhs );
                }
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                const LInt last_l = rp[i_end];

                const LInt look_ahead = static_cast<LInt>(
                    Tools::Max(
                        Size_T(1),
                        CacheLineWidth
                        /
                        (sizeof(X_T) * Tools::Max(static_cast<Size_T>(nrhs),NRHS))
                    )
                );
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

                    if( l_end > l_begin )
                    {
                        // Overwrite for first element in row.
                        {
                            const LInt l = l_begin;
                            const Int  j = ci[l] - base;

                            if constexpr ( prefetchQ )
                            {
                                // This prefetch would cause segfaults without the check.
                                if( l + look_ahead < last_l )
                                {
                                    prefetch( &X[ldX * (ci[l + look_ahead] - base)], 0, 0 );
                                }
                            }
                            
                            // We use this `if constexpr` here so that we do not read from a when it is a nullptr
                            if constexpr ( a_flag == F_T::Generic )
                            {
                                combine_buffers<a_flag,F_T::Zero,NRHS,Sequential>(
                                    static_cast<a_T>(a[l]), &X[ldX * j],
                                    Scalar::Zero<z_T>,      &z[0],
                                    nrhs
                                );
                            }
                            else if constexpr ( a_flag == F_T::Plus )
                            {
                                combine_buffers<a_flag,F_T::Zero,NRHS,Sequential>(
                                    Scalar::One <a_T>, &X[ldX * j],
                                    Scalar::Zero<z_T>, &z[0],
                                    nrhs
                                );
                            }
                        }
                        
                        // Add remaining entries.
                        for( LInt l = l_begin + LInt(1); l < l_end; ++l )
                        {
                            const Int j = ci[l] - base;

                            if constexpr ( prefetchQ )
                            {
                                // This prefetch would cause segfaults without the check.
                                if( l + look_ahead < last_l )
                                {
                                    prefetch( &X[ldX * (ci[l + look_ahead]- base)], 0, 0 );
                                }
                            }

                            // Add-in
                            if constexpr ( a_flag == F_T::Generic )
                            {
                                combine_buffers<F_T::Generic,F_T::Plus,NRHS,Sequential>(
                                    static_cast<a_T>(a[l]), &X[ldX * j],
                                    Scalar::One<z_T>,       &z[0],
                                    nrhs
                                );
                            }
                            else if constexpr ( a_flag == F_T::Plus )
                            {
                                // We use if Scalar::One<a_T> here so that we do not read from a when it is a nullptr
                                combine_buffers<F_T::Plus,F_T::Plus,NRHS,Sequential>(
                                    Scalar::One<a_T>, &X[ldX * j],
                                    Scalar::One<z_T>, &z[0],
                                    nrhs
                                );
                            }
                        }
                        
                        // Incorporate the local updates into Y-buffer.
                        combine_buffers<alpha_flag,beta_flag,NRHS,Sequential>(
                            alpha_, &z[0],
                            beta_,  &Y[ldY * i],
                            nrhs
                        );
                    }
                    else // if( l_end == l_begin )
                    {
                        // Modify the relevant portion of the Y-buffer.
                        if constexpr( beta_flag == F_T::Zero )
                        {
                            zerofy_buffer<NRHS,Sequential>( &Y[ldY * i], nrhs );
                        }
                        else if constexpr( beta_flag == F_T::Generic )
                        {
                            scale_buffer<NRHS,Sequential>( beta_, &Y[ldY * i], nrhs );
                        }
                        else if constexpr( beta_flag == F_T::Plus )
                        {
                            // Do nothing.
                        }
                    }
                }
            },
            job_ptr.ThreadCount()
        );
    }

#if( __has_attribute(__ext_vector_type__) )

    template<
        F_T a_flag, F_T alpha_flag, F_T beta_flag,
        Size_T NRHS = VarSize, bool base,
        typename alpha_T, typename X_T, typename beta_T, typename Y_T
    >
    void SpMM_vec(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<X_T> X, const Int ldX,
        cref<beta_T>  beta,  mptr<Y_T> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        (void)m;
        (void)n;
    
        std::string tag = std::string(ClassName()+"::SpMM_vec")
            + "<" + ToString(a_flag)
            + "," + ToString(alpha_flag)
            + "," + ToString(beta_flag)
            + "," + (NRHS <= 0 ? std::string("VarSize") : ToString(NRHS))
            + "," + ToString(base)
            + "," + TypeName<alpha_T>
            + "," + TypeName<X_T>
            + "," + TypeName<beta_T>
            + "," + TypeName<Y_T>
            +">(" + ToString(ldX)
            + "," + ToString(ldY)
            + ")";
        
        TOOLS_PTIMER(timer,tag);
        
        static_assert(NRHS!=0, "SpMM_vec only implements static size behavior.");
        
        // Only to be called by SpMM_impl which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == One
        //  - alpha_flag == One
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic

        // Treats sparse matrix as a binary matrix if a_flag != F_T::Generic.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.
        
        // TODO: Use real types for alpha, beta, x, a, y if applicable.
        // TODO: Maybe use simple combine_buffer to merge y and z.
        
        using y_T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<X_T>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        using x_T = y_T;
        
        if constexpr ( a_flag == F_T::Generic )
        {
            if ( a == nullptr )
            {
                eprint( tag + ": a_flag == F_T::Generic, but the pointer a is a nullptr. Aborting.");
                return;
            }
        }
        
        constexpr bool prefetchQ = true;
//        constexpr bool prefetchQ = false;
        
        
        constexpr LInt look_ahead = static_cast<LInt>(
            Tools::Max(
                Size_T(1),
                CacheLineWidth / (sizeof(Scal) * NRHS)
            )
        );
             
        ParallelDo(
            [&]( const Int thread )
            {
                vec_T<NRHS,x_T> x;
     
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                const LInt last_l = rp[i_end];

                for( Int i = i_begin; i < i_end; ++i )
                {
                    vec_T<NRHS,y_T> y ( Scalar::Zero<y_T> );
                    
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

                    if( l_end > l_begin )
                    {
                        // Add for first entry.
                        for( LInt l = l_begin; l < l_end; ++l )
                        {
                            const Int j = ci[l] - base;

                            if constexpr ( prefetchQ )
                            {
                                // This prefetch would cause segfaults without the check.
                                if( l + look_ahead < last_l )
                                {
                                    prefetch( &X[ldX * (ci[l + look_ahead] - base)], 0, 0 );
                                }
                            }
                            
                            copy_buffer<NRHS>( &X[ldX * j], get_ptr(x) );

                            if constexpr ( a_flag == F_T::Generic )
                            {
                                y += a[l] * x;
                            }
                            else if constexpr ( a_flag == F_T::Plus )
                            {
                                y += x;
                            }
                        }
                        
                        // Incorporate the local updates into Y-buffer.
                                   
                        combine_buffers<alpha_flag,beta_flag,NRHS>(
                            alpha, get_ptr(y), beta, &Y[ldY * i]
                        );
                    }
                    else
                    {
                        // Modify the relevant portion of the Y-buffer.
                        if constexpr( beta_flag == F_T::Zero )
                        {
                            zerofy_buffer<NRHS,Sequential>( &Y[ldY * i] );
                        }
                        else if constexpr( beta_flag == F_T::Plus )
                        {
                            // Do nothing.
                        }
                        else if constexpr( beta_flag == F_T::Generic )
                        {
                            scale_buffer<NRHS,Sequential>( beta, &Y[ldY * i] );
                        }
                    }
                }
            },
            job_ptr.ThreadCount()
        );
    }
#endif // ( __has_attribute(__ext_vector_type__) )
