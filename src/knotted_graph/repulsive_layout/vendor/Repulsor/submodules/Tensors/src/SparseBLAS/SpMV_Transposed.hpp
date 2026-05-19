public:

    template<typename a_T, typename X_T, typename b_T, typename Y_T>
    void SpMV_Transposed(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<a_T> alpha_, cptr<X_T> X,
        cref<b_T> beta_,  mptr<Y_T> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        // TODO: Improve load balancing.
        
        // This is basically a large switch to determine at runtime, which instantiation of SpMV_Transposed_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMV_Transposed_impl are instantiated.
        
        using alpha_T = std::conditional_t< Scalar::RealQ<a_T>, Scalar::Real<Y_T>, Y_T>;
        using beta_T  = std::conditional_t< Scalar::RealQ<b_T>, Scalar::Real<Y_T>, Y_T>;
        
//        StaticParameterCheck<alpha_T,X_T,beta_T,Y_T>();
        
        const alpha_T alpha = ( rp[m] > LInt(0) ) ? scalar_cast<Y_T>(alpha_) : scalar_cast<a_T>(0);
        const beta_T  beta  = scalar_cast<Y_T>(beta_);
        
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if ( alpha == static_cast<alpha_T>(0) )
        {
            if ( beta == static_cast<beta_T>(0) )
            {
                zerofy_buffer( Y, m, job_ptr.ThreadCount() );
            }
            else if ( beta == static_cast<beta_T>(1) )
            {
                // Do nothing.
            }
            else
            {
                scale_buffer( beta, Y, m, job_ptr.ThreadCount() );
            }
            return;
        }
        
        auto job = [=,this]<F_T a_flag, F_T alpha_flag, F_T beta_flag>()
        {
            this->SpMV_Transposed_impl<a_flag,alpha_flag,beta_flag>(
                rp,ci,a,m,n,alpha,X,beta,Y,job_ptr
            );
        };
        
        if( a != nullptr )
        {
            constexpr F_T a_flag = F_T::Generic;
            
            if( alpha == static_cast<alpha_T>(1) )
            {
                constexpr F_T alpha_flag = F_T::Plus;
                
                if( beta == static_cast<beta_T>(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == static_cast<beta_T>(1) )
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
                
                if( beta == static_cast<beta_T>(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == static_cast<beta_T>(0) )
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
            
            if( alpha == static_cast<alpha_T>(1) )
            {
                constexpr F_T alpha_flag = F_T::Plus;
                
                if( beta == static_cast<beta_T>(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == static_cast<beta_T>(1) )
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
                
                if( beta == static_cast<beta_T>(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == static_cast<beta_T>(0) )
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

    template<F_T a_flag, F_T alpha_flag, F_T beta_flag, typename a_T, typename X_T, typename b_T, typename Y_T>
    void SpMV_Transposed_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<a_T> alpha, cptr<X_T>  x,
        cref<b_T> beta,  mptr<Y_T> y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        (void)n;
        
        std::string tag = std::string(ClassName()+"::SpMV_Transposed_impl<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +TypeName<a_T>+","
            +TypeName<X_T >+","
            +TypeName<b_T>+","
            +TypeName<Y_T>+">";
        
        TOOLS_PTIMER(timer,tag);
        
        // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == One
        //  - alpha_flag == One
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic
        
        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.

        using T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<X_T>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        
//        if constexpr ( beta_flag == F_T::Zero )
//        {
//            zerofy_buffer<VarSize,Parallel>( y, n );
//        }
//        else if constexpr ( beta_flag == F_T::Plus )
//        {
//            // Do nothing
//        }
//        else // beta_flag == F_T::Generic or beta_flag == F_T::Minus
//        {
//            scale_buffer<VarSize,Parallel>( beta, y, n );
//        }
//
        ParallelDo(
            [&]( const Int thread )
            {
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

//                const LInt last_l = rp[i_end];

//                constexpr LInt look_ahead = 1;

                if constexpr ( beta_flag == F_T::Zero )
                {
                    zerofy_buffer<VarSize,Sequential>( &y[i_begin], i_end - i_begin );
                }
                else if constexpr ( beta_flag == F_T::Plus )
                {
                    // Do nothing
                }
                else // beta_flag == F_T::Generic or beta_flag == F_T::Minus
                {
                    scale_buffer<VarSize,Sequential>( beta, &y[i_begin], i_end - i_begin );
                }

                for( Int j = 0; j < m; ++j )
                {
                    const LInt l_begin = rp[j  ];
                    const LInt l_end   = rp[j+1];

                    if( l_begin < l_end )
                    {
                        LInt l = l_begin;

                        while( l < l_end )
                        {
                            const Int  i = ci[l];

                            if ( i >= i_begin )
                            {
                                break;
                            }

                            ++l;
                        }

                        while( l < l_end )
                        {
                            const Int  i = ci[l];

                            if ( i >= i_end )
                            {
                                break;
                            }

                            Scal product;

                            if constexpr ( a_flag == F_T::Generic )
                            {
                                product = scalar_cast<T>(a[l]) * scalar_cast<T>(x[i]);
                            }
                            else
                            {
                                product = scalar_cast<T>(x[i]);
                            }

                            combine_scalars<alpha_flag,F_T::Plus>( alpha, product, Scalar::One<Scal>, y[j] );

                            ++l;
                        }
                    }
                }
            },
            job_ptr.ThreadCount()
        );
    }

