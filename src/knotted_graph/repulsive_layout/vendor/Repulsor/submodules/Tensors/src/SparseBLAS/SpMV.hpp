public:

    template<bool base, typename alpha_T, typename x_t, typename beta_T, typename y_t>
    void SpMV(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<x_t> x,
        cref< beta_T> beta,  mptr<y_t> y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMV_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMV_impl are instantiated.
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if ( alpha == alpha_T(0) )
        {
            if ( beta == beta_T(0) )
            {
                zerofy_buffer( y, m, job_ptr.ThreadCount() );
            }
            else if ( beta == beta_T(1) )
            {
                // Do nothing.
            }
            else
            {
                scale_buffer( beta, y, m, job_ptr.ThreadCount() );
            }
            return;
        }
        
        auto job = [=,this]<F_T a_flag, F_T alpha_flag, F_T beta_flag>()
        {
            this->SpMV_impl<a_flag,alpha_flag,beta_flag,base>(
                rp,ci,a,m,n,alpha,x,beta,y,job_ptr
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
        else
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
        Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag, bool base = 0,
        typename alpha_T, typename x_T, typename  beta_T, typename y_T>
    void SpMV_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<x_T> x,
        cref< beta_T> beta,  mptr<y_T> y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        (void)m;
        (void)n;
        
        std::string tag = std::string(ClassName()+"::SpMV_impl")
            + "<" + ToString(a_flag)
            + "," + ToString(alpha_flag)
            + "," + ToString(beta_flag)
            + "," + ToString(base)
            + "," + TypeName<alpha_T>
            + "," + TypeName<x_T>
            + "," + TypeName<beta_T>
            + "," + TypeName<y_T>
            + ">";
        
        TOOLS_PTIMER(timer,tag);
        
        // TODO: Add better check for pointer overlap.
        
        if constexpr ( std::is_same_v<x_T,y_T> )
        {
            if( x == y )
            {
                eprint( tag + ": Input and output pointer coincide. This is not safe. Aborting.");
                return;
            }
        }
        
        // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == Plus
        //  - alpha_flag == Plus
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic
        
        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.

        
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
            Scalar::Complex<x_T>,
            Scalar::Real<x_T>
        >;
        
        // Define z_T so that it can hold a * x.
        using z_T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<x_T>,
            typename Scalar::Complex<x_T>,
            typename Scalar::Real<x_T>
        >;

        // Check whether z = a * x will work.
//        StaticParameterCheck<Scalar::Real<z_T>,z_T,a_T,x_T>();

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
            Scalar::Complex<y_T>,
            Scalar::Real<y_T>
        >;
        
        const alpha_T_ alpha_ = static_cast<alpha_T_>(alpha);
        const beta_T_  beta_  = static_cast<beta_T_ >(beta );
        
        // Check whether y = alpha_ * z + beta_ * y will work.
//        StaticParameterCheck<alpha_T_,z_T,beta_T_,y_T>();
        
        ParallelDo(
            [&]( const Int thread )
            {
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                const LInt last_l = rp[i_end];
                
                constexpr LInt look_ahead = 1;

                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

                    z_T z ( 0 );
                    
                    for( LInt l = l_begin; l < l_end; ++l )
                    {
                        const Int j = ci[l] - base;
    
                        // This prefetch would cause segfaults without the check.
                        if( l + look_ahead < last_l )
                        {
                            prefetch( &x[ci[l + look_ahead] - base], 0, 0 );
                        }
    
                        if constexpr ( a_flag == F_T::Generic )
                        {
                            z += static_cast<a_T>(a[l]) * static_cast<z_T>(x[j]);
                        }
                        else
                        {
                            z += static_cast<z_T>(x[j]);
                        }
                    }
                    
                    combine_scalars<alpha_flag,beta_flag>( alpha_, z, beta_, y[i] );
                }
            },
            job_ptr.ThreadCount()
        );
    }

