#pragma once

#include "Tensors.hpp"

#define TENSORS_CONJUGATEGRADIENT_H

namespace Tensors
{
    // EQ_COUNT = number of right hand sides.
    // If you know this and compile time, then enter it in the template.
    // If you don't know this at compile time, then use EQ_COUNT = VarSize (==0) and specify
    // the eq_count_ in the constructor.
    
    // Scal_ is the floating point type that is used internally.
    
    template<Size_T NRHS_, typename Scal_, IntQ Int_,
        bool A_verboseQ = true, bool P_verboseQ = true
    >
    class ConjugateGradient final
    {
    public:
        
        using Scal     = Scal_;
        using Real     = Scalar::Real<Scal>;
        using Int      = Int_;
        
        static constexpr Int NRHS = int_cast<Int>(NRHS_);
        
        using RealVector_T = Tensor1<Real,Int>;

    protected:
        
        const Int n;
        const Int max_iter;
        const Int nrhs;
        const Int thread_count;
        
        Tensor2<Scal,Int> r;
        Tensor2<Scal,Int> u;
        Tensor2<Scal,Int> p;
        Tensor2<Scal,Int> x;
        Tensor2<Scal,Int> z;
        
        ThreadTensor2<Real,Int> reduction_buffer;
        
        RealVector_T TOL;
        RealVector_T b_squared_norms;
        
        RealVector_T alpha;
        RealVector_T beta;
        RealVector_T rho;
        RealVector_T rho_old;
        
        JobPointers<Int> job_ptr;
        
        Int iter = 0;
        
        Real time_elapsed       = 0;
        Real relative_tolerance = 0.0001;
        bool use_initial_guessQ = false;
        
    public:
        
        ConjugateGradient(
            const Int n_,
            const Int max_iter_,
            const Size_T eq_count_ = NRHS,
            const Size_T thread_count_ = 1
        )
        :   n               ( n_                                    )
//        ,   max_iter        ( Min(max_iter_,n)                      )
        ,   max_iter        ( Max(Int(0),max_iter_)                 )
        ,   nrhs            ( ( NRHS > VarSize ? NRHS : static_cast<Int>(eq_count_ ) )  )
        ,   thread_count    ( static_cast<Int>(thread_count_)       )
        ,   r               ( n, nrhs                               )
        ,   u               ( n, nrhs                               )
        ,   p               ( n, nrhs                               )
        ,   x               ( n, nrhs                               )
        ,   z               ( n, nrhs                               )
        ,   reduction_buffer( thread_count, nrhs                    )
        ,   TOL             ( nrhs                                  )
        ,   b_squared_norms ( nrhs                                  )
        ,   alpha           ( nrhs                                  )
        ,   beta            ( nrhs                                  )
        ,   rho             ( nrhs                                  )
        ,   rho_old         ( nrhs                                  )
        ,   job_ptr         ( n, thread_count                       )
        {}
        
        // Default constructor
        ConjugateGradient() = delete;
        // Destructor
        ~ConjugateGradient() = default;
        // Copy constructor
        ConjugateGradient( const ConjugateGradient & other ) = default;
        // Copy assignment operator
        ConjugateGradient & operator=( const ConjugateGradient & other ) = default;
        // Move constructor
        ConjugateGradient( ConjugateGradient && other ) = default;
        // Move assignment operator
        ConjugateGradient & operator=( ConjugateGradient && other ) = default;
        
        // Computes X_inout <- a * A^{-1} . B_in + b * X_inout via CG method.
        // Uses P as precondition, i.e., P should be a proxy of A^{-1}.
        template<
            typename Operator_T, typename Preconditioner_T,
            typename a_T, typename B_T, typename b_T, typename X_T
        >
        bool operator()(
            mref<Operator_T>       A,
            mref<Preconditioner_T> P,
            const a_T a, cptr<B_T> B_in,    const Int ldB,
            const b_T b, mptr<X_T> X_inout, const Int ldX,
            const Real relative_tolerance_,
            // We force the use to actively request that the initial guess is used,
            // because this is a common source of bugs.
            const bool use_initial_guessQ_ = false
        )
        {
            // `A` and `P` must be a functions or lambda with prototypes
            //
            // `A( cptr<Scal> v, mptr<Scal> w)`
            //
            // and
            //
            // `P( cptr<Scal> v, mptr<Scal> w)`
            //
            // They may return something, but the returned value is ignored.
            
            std::string tag = ClassName()+"::operator<" + TypeName<B_T> + "," + TypeName<X_T> + ">()";
            
            TOOLS_PTIMER(timer,tag);
            
            Time start_time = Clock::now();
            
            use_initial_guessQ = use_initial_guessQ_;
            relative_tolerance = relative_tolerance_;

            if( use_initial_guessQ && (b != b_T(0)) )
            {
                wprint( tag + ": use_initial_guessQ == true and b != 0. Typically, this does not make sense." );
            }
            
            iter = 0;
            bool succeeded = false;
            
            if( a == a_T(0) )
            {
                wprint(tag + ": Factor a is 0. Returning b * X_inout");

                scale_matrix<VarSize,NRHS,Parallel>(
                    b, X_inout, ldX, n, nrhs, thread_count
                );
                
                succeeded = true;

                logvalprint( tag + " iter"      , iter      );
                logvalprint( tag + " succeeded" , succeeded );
                
                TOOLS_PTOC(tag);
                
                return succeeded;
            }
            
            // r = b
            r.Read( B_in, ldB, thread_count );
            
            TOOLS_PTIC(ClassName()+": Compute norm of right hand side.");
            
            // z = P.b
            ApplyPreconditioner(P,r,z);
            
            // rho = r.z
            ComputeScalarProducts( r, z, rho );
            
            Real factor = relative_tolerance * relative_tolerance;
            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
            {
                b_squared_norms[k] = std::abs(rho[k]);
                TOL[k] = b_squared_norms[k] * factor;
            }
            
            TOOLS_PTOC(ClassName()+": Compute norm of right hand side.");
            
            if( b_squared_norms.CountNaNs() > Int(0) )
            {
                eprint(tag + ": Right-hand side contains NaNs. Doing nothing.");
                
                succeeded = false;

                logvalprint( tag + " iter"      , iter      );
                logvalprint( tag + " succeeded" , succeeded );
                
                return succeeded;
            }
            
            if( b_squared_norms.Max() <= 0 )
            {
                wprint(tag + ": Right-hand side is 0. Returning b * X_inout.");

                scale_matrix<VarSize,NRHS,Parallel>(
                    b, X_inout, ldX, n, nrhs, thread_count
                );
                
                succeeded = true;

                logvalprint( tag + " iter"      , iter      );
                logvalprint( tag + " succeeded" , succeeded );
                
                return succeeded;
            }
                
            // r = b - A.x
            if( use_initial_guessQ )
            {
                logprint( tag + ": Input x_inout is nonzero. Using it as initial guess." );
                
                x.Read( X_inout, ldX, thread_count );
                
                // u = A.x
                ApplyOperator(A,x,u);
    
                // r = b - A.x
                Do<Parallel>(
                    [this]( const Int i )
                    {
                        mptr<Scal> r_i = r.data(i);
                        cptr<Scal> u_i = u.data(i);
                        
                        for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                        {
//                            r[i][k] -= u[i][k];
                            r_i[k] -= u_i[k];
                        }
                    },
                    n, thread_count
                );
            }
            
            // z = P.r
            ApplyPreconditioner(P,r,z);
            
            p.Read( z.data(), nrhs, thread_count );
            
            // rho = r.z
            ComputeScalarProducts( r, z, rho );
            
            succeeded = CheckResiduals();

            while( !succeeded && (iter < max_iter ) )
            {
                // u = A.p
                ApplyOperator(A,p,u);
                
                // alpha = rho / (p.u);
                ComputeScalarProducts( p, u, alpha );
                for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                {
                    alpha[k] = rho[k] / alpha[k];
                }
                
                if( (iter == 0) && (!use_initial_guessQ) )
                {
                    // x = 0 + alpha p;
                    // r = r - alpha u;
                    
                    Do<Parallel>(
                        [this]( const Int i )
                        {
                            mptr<Scal> x_i = x.data(i);
                            mptr<Scal> r_i = r.data(i);
                            
                            cptr<Scal> p_i = p.data(i);
                            cptr<Scal> u_i = u.data(i);
                            
                            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                            {
                                x_i[k]  = alpha[k] * p_i[k];
                                r_i[k] -= alpha[k] * u_i[k];
                            }
                        },
                        n, thread_count
                    );
                }
                else
                {
                    Do<Parallel>(
                        [this]( const Int i )
                        {
                            mptr<Scal> x_i = x.data(i);
                            mptr<Scal> r_i = r.data(i);
                            
                            cptr<Scal> p_i = p.data(i);
                            cptr<Scal> u_i = u.data(i);
                            
                            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                            {
                                x_i[k] += alpha[k] * p_i[k];
                                r_i[k] -= alpha[k] * u_i[k];
                            }
                        },
                        n, thread_count
                    );
                }
                
                // z = P.r;
                ApplyPreconditioner(P,r,z);
                
                // rho_old = rho
                swap( rho_old, rho );
                
                // rho = r.z;
                ComputeScalarProducts( r, z, rho );
                
                // beta = rho / rho_old;
                for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                {
                    beta[k] = rho[k] / rho_old[k];
                }
                
                // TODO: Put this at the start of the while loop, and only for iter > 0?
                // p = z + beta p;
                Do<Parallel>(
                    [this]( const Int i )
                    {
                        mptr<Scal> p_i = p.data(i);
                        cptr<Scal> z_i = z.data(i);
                        
                        for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                        {
                            p_i[k] = z_i[k] + beta[k] * p_i[k];
                        }
                    },
                    n, thread_count
                );
                
                succeeded = CheckResiduals();
                ++iter;
            }
            
            
            combine_matrices_auto<VarSize,NRHS,Parallel>(
                scalar_cast<X_T>(a), x.data(), nrhs,
                scalar_cast<X_T>(b), X_inout,  ldX,
                n, nrhs, thread_count
            );
            
            time_elapsed = Tools::Duration( start_time, Clock::now() );
            
            logprint( Stats() );
            
            return succeeded;
        }
        
        
    protected:
        
        template<typename Operator_T>
        void ApplyOperator(
            mref<Operator_T> A, cref<Tensor2<Scal,Int>> X, mref<Tensor2<Scal,Int>> Y
        )
        {
            if constexpr ( A_verboseQ )
            {
                TOOLS_PTIC(ClassName()+ "::ApplyOperator");
            }
            
            (void)A( X.data(), Y.data() );
            
            if constexpr ( A_verboseQ )
            {
                TOOLS_PTOC(ClassName()+ "::ApplyOperator");
            }
        }
        
        template<typename Preconditioner_T>
        void ApplyPreconditioner(
            mref<Preconditioner_T> P, cref<Tensor2<Scal,Int>> v, mref<Tensor2<Scal,Int>> w
        )
        {
            if constexpr ( P_verboseQ )
            {
                TOOLS_PTIC(ClassName()+ "::ApplyPreconditioner");
            }
            
            (void)P( v.data(), w.data() );
            
            
            if constexpr ( P_verboseQ )
            {
                TOOLS_PTOC(ClassName()+ "::ApplyPreconditioner");
            }
        }
        
        void ComputeScalarProducts(
            mref<Tensor2<Scal,Int>> v,
            mref<Tensor2<Scal,Int>> w,
            mref<RealVector_T> dots
        )
        {
//            TOOLS_PTIMER(timer,ClassName()+ "::ComputeScalarProducts");
            
            ParallelDo(
                [this,&v,&w]( const Int thread )
                {
                    auto & sums = reduction_buffer[thread];
                    
                    sums.SetZero();
                    
                    const Int i_begin = job_ptr[thread    ];
                    const Int i_end   = job_ptr[thread + 1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        cptr<Scal> v_i = v.data(i);
                        cptr<Scal> w_i = w.data(i);
                        
                        for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                        {
                            // We know that all scalar products that we compute have to be real-valued.
//                            sums[k] += Re(Conj(v[i][k]) * w[i][k]);
                            
                            if constexpr ( Scalar::ComplexQ<Scal> )
                            {
                                sums[k] += Re(v_i[k]) * Re(w_i[k]) + Im(v_i[k]) * Im(w_i[k]) ;
                            }
                            else
                            {
                                sums[k] += v_i[k] * w_i[k];
                            }
                        }
                    }
                },
                thread_count
            );
            
            reduction_buffer.AddReduce( dots.data(), false );
        }
        
    public:
        
        Int IterationCount() const
        {
            return iter;
        }
        
        RealVector_T Residuals() const
        {
            RealVector_T res (nrhs);
            
            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
            {
                res[k] = Sqrt( Abs(rho[k]) );
            }
            
            return res;
        }
        
        RealVector_T RelativeResiduals() const
        {
            RealVector_T res(nrhs);
            
            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
            {
                res[k] = Sqrt( Abs(rho[k]) / b_squared_norms[k] );
            }
            return res;
        }
        
        bool CheckResiduals() const
        {
            bool succeeded = true;
            
            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
            {
                succeeded = succeeded && ( Abs(rho[k]) <= TOL[k]);
            }
            
            return succeeded;
        }
        
        std::string Stats() const
        {
            return std::string()
            + "\n==== " + ClassName() + " Stats ====\n\n"
            + " time_elapsed       = " + ToString(time_elapsed) + "\n"
            + " n                  = " + ToString(n) + "\n"
            + " nrhs               = " + ToString(nrhs) + "\n"
            + " iter               = " + ToString(iter)     + "\n"
            + " max_iter           = " + ToString(max_iter) + "\n"
            + " relative_tolerance = " + ToString(relative_tolerance) + "\n"
            + " use_initial_guessQ = " + ToString(use_initial_guessQ) + "\n"
            + "\n==== " + ClassName() + " Stats ====\n\n"
            + " relative residuals = " + OutString::FromVector( RelativeResiduals().data(), nrhs ) + "\n";
        }
        
        std::string ClassName() const
        {
            return std::string( "ConjugateGradient")
                + "<" + (NRHS <= VarSize ? std::string("VarSize") : ToString(NRHS) )
                + "," + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + ToString(A_verboseQ)
                + "," + ToString(P_verboseQ)
                +">(" + ToString(nrhs)
                + ")";
        }
        
    }; // class ConjugateGradient
        
        
    
        
} // namespace Tensors


