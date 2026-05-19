#pragma once

namespace Tools
{
    
    template<typename Real, typename UInt>
    class uniform_dist
    {
        static_assert(Scalar::RealQ<Real>,"");
        
    private:
        
        const Real a     = 0;
        const Real b     = 1;
        
        const Real scale = 1;
                
    public:
        
        uniform_dist( const Real a_, const Real b_ ) noexcept
        : a(a_)
        , b(b_)
        , scale( (b-a) / std::numeric_limits<UInt>::max() )
        {}
        
        ~uniform_dist() = default;
        
        template<typename R>
        TOOLS_FORCE_INLINE Real operator()( R & r ) const
        {
            return a + static_cast<Real>(r()) * scale;
        }
    };
    
    template<typename Real, typename UInt>
    class fake_gaussian_dist
    {
        static_assert(Scalar::RealQ<Real>,"");
        
        static constexpr Real scale = Scalar::One<Real> / (UInt64(1) << 20);
        
    public:
        
        fake_gaussian_dist() = default;
        
        ~fake_gaussian_dist() = default;
        
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Real operator()( PRNG_T & random_engine ) const
        {
            const UInt r = random_engine();
            
            return ((r & 0x1fffff) + ((r >> 21) & 0x1fffff) + ((r >> 42) & 0x1fffff)) * scale - Scalar::Three<Real>;
        }
    };
    
    
    template<typename Real, typename UInt>
    class BoxMuller
    {
        static_assert(Scalar::RealQ<Real>,"");
        
    private:
        
        Real X;
        Real Y;
        
        uniform_dist<Real,UInt64> r_dist { Scalar::One<Real>, Scalar::Zero<Real> };
        uniform_dist<Real,UInt64> phi_dist { -Scalar::Pi<Real>, Scalar::Pi<Real> };
        
        bool computedQ = 0;
        
        
    public:
        
        BoxMuller() = default;
        
        ~BoxMuller() = default;
                
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE void operator()( PRNG_T & random_engine, Real & x, Real & y )
        {
            TOOLS_MAKE_FP_FAST()
            
            const Real r   = Sqrt(-Scalar::Two<Real> * std::log( r_dist(random_engine)) );
            
            const Real phi = phi_dist(random_engine);
            
            x = r * std::cos(phi);
            y = r * std::sin(phi);
        }
        
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Real operator()( PRNG_T & random_engine )
        {
            TOOLS_MAKE_FP_FAST()
            
            if( computedQ )
            {
                computedQ = false;
                return Y;
            }
            else
            {
                const Real r   = Sqrt( -Scalar::Two<Real> * std::log( r_dist(random_engine)) );
                
                const Real phi = phi_dist(random_engine);
                
                X = r * std::cos(phi);
                Y = r * std::sin(phi);
                
                computedQ = true;
                
                return X;
            }
        }
        
    };
    
    template<typename Real, typename UInt>
    class Gaussian
    {
        static_assert(Scalar::RealQ<Real>,"");
        
    private:
        
        Real X;
        Real Y;
        Real r2;
        
        uniform_dist<Real,UInt64> dist { -Scalar::One<Real>, Scalar::One<Real> };
        
        bool computedQ = 0;
        
    public:
        
        Gaussian() = default;
        
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Real operator()( PRNG_T & random_engine )
        {
            if( computedQ )
            {
                computedQ = false;
                return Y;
            }
            else
            {
                TOOLS_MAKE_FP_FAST()
                
                X = dist(random_engine);
                Y = dist(random_engine);
                
                r2 = X * X + Y * Y;
                
                while ( r2 > Scalar::One<Real> || r2 == Scalar::Zero<Real> )
                {
                    X = dist(random_engine);
                    Y = dist(random_engine);
                    
                    r2 = X * X + Y * Y;
                }
                
                const Real r  = Sqrt( - Scalar::Two<Real> * std::log( r2 ) / r2 );
                
                X *= r;
                Y *= r;
                
                computedQ = true;
                
                return X;
            }
        }
        
    };
    
    /*!brief Wrapped Gaussian distribution with center `mu`, standard deviation `sigma`, and period `L`. Returns values in the half-open interval [0,L).
     */
    template<typename Real>
    class WrappedGaussianDistribution
    {
        static_assert(Scalar::FloatQ<Real>,"");
        static_assert(Scalar::RealQ<Real>,"");
        
    private:
        
        Real mu     = 0;
        Real sigma  = 1;
        Real L      = Scalar::TwoPi<Real>;
        Real L_inv  = Inv(Scalar::TwoPi<Real>);
        
        std::normal_distribution<Real> gaussian {mu,sigma};
        
    public:
        
        WrappedGaussianDistribution() = default;
        
        WrappedGaussianDistribution(
            const Real center,
            const Real standard_deviation,
            const Real period
        )
        :   mu       { center }
        ,   sigma    { standard_deviation }
        ,   L        { period }
        ,   L_inv    { Inv<Real>(period) }
        ,   gaussian { mu, standard_deviation }
        {}
  
        Real Mod( const Real x )
        {
            return x - L * std::floor(x * L_inv);
        }
        
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Real operator()( PRNG_T & random_engine )
        {
            return Mod(gaussian(random_engine));
        }
        
        Real Center() const
        {
            return mu;
        }
        
        Real StandardDeviation() const
        {
            return sigma;
        }
        
        Real Period() const
        {
            return L;
        }
    };
    
    
    /*!brief Discrete wrapped Gaussian distribution with center `mu`, standard deviation `sigma`, and period `L`. Returns values in the half-open interval [0,L).
     */
    template<IntQ Int, FloatQ Real>
    class DiscreteWrappedGaussianDistribution
    {
    private:
        
        WrappedGaussianDistribution<Real> wrapped_gaussian;
        
        Int n;
        
    public:
        
        DiscreteWrappedGaussianDistribution() = delete;
        
        DiscreteWrappedGaussianDistribution(
            const Real center,
            const Real standard_deviation,
            const Int  period
        )
        :   wrapped_gaussian { center, standard_deviation, Real(period) }
        ,   n                { period }
        {}
  
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Int operator()( PRNG_T & random_engine )
        {
            return static_cast<Int>(std::floor(wrapped_gaussian(random_engine)));
        }
        
        Real Center() const
        {
            return wrapped_gaussian.Center();
        }
        
        Real StandardDeviation() const
        {
            return wrapped_gaussian.StandardDeviation();
        }
        
        Int Period() const
        {
            return n ;
        }
    };
    
    
    
    
    /*!brief Wrapped Laplace distribution with center `mu`, scale parameter `beta`, and period `L`. Returns values in the half-open interval [0,L).  Here the Laplace distribution is given by f(x) = 1/(2 * beta) * exp( - |x-mu|/beta ).
     */
    template<typename Real>
    class WrappedLaplaceDistribution
    {
        static_assert(Scalar::FloatQ<Real>,"");
        static_assert(Scalar::RealQ<Real>,"");
        
    private:
        
        Real mu     = 0;
        Real beta   = 1;
        Real L      = Scalar::TwoPi<Real>;
        Real L_inv  = Inv(Scalar::TwoPi<Real>);
        
        std::exponential_distribution<Real> exp_dist { Inv<Real>(beta) };
        std::uniform_int_distribution<int>  coin     { 0, 1 };
        
    public:
        
        WrappedLaplaceDistribution() = default;
        
        WrappedLaplaceDistribution(
            const Real center,
            const Real scale_parameter,
            const Real period
        )
        :   mu       { center }
        ,   beta     { scale_parameter }
        ,   L        { period }
        ,   L_inv    { Inv<Real>(period) }
        ,   exp_dist { Inv<Real>(scale_parameter) }
        {}
  
        Real Mod( const Real x )
        {
            return x - L * std::floor(x * L_inv);
        }
        
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Real operator()( PRNG_T & random_engine )
        {
            const Real sign = (coin(random_engine) ? Real(1) : Real(-1) );
            return Mod( mu + sign * exp_dist(random_engine));
        }
        
        Real Center() const
        {
            return mu;
        }
        
        Real ScaleParameter() const
        {
            return beta;
        }
        
        Real Period() const
        {
            return L;
        }
    };
    
    /*!brief Discrete wrapped Laplace distribution with center `mu`, scale parametr `beta`, and period `L`. Returns values in the half-open interval [0,L).
     */
    template<IntQ Int, FloatQ Real>
    class DiscreteWrappedLaplaceDistribution
    {
        
    private:
        
        WrappedLaplaceDistribution<Real> wrapped_laplacian;
        
        Int n;
        
    public:
        
        DiscreteWrappedLaplaceDistribution() = delete;
        
        DiscreteWrappedLaplaceDistribution(
            const Real center,
            const Real beta,
            const Int  period
        )
        :   wrapped_laplacian { center, beta, Real(period) }
        ,   n                 { period }
        {}
  
        template<typename PRNG_T>
        TOOLS_FORCE_INLINE Int operator()( PRNG_T & random_engine )
        {
            return static_cast<Int>(std::floor(wrapped_laplacian(random_engine)));
        }
        
        Real Center() const
        {
            return wrapped_laplacian.Center();
        }
        
        Real ScaleParameter() const
        {
            return wrapped_laplacian.ScaleParameter();
        }
        
        Int Period() const
        {
            return n ;
        }
    };

}

 


