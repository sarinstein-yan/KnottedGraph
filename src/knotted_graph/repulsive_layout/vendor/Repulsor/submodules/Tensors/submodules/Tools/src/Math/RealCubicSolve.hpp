#pragma once

namespace Tools
{
    
    template<typename Real>
    int RealCubicSolve_UnitInterval_RegulaFalsi(
        const Real c_3, const Real c_2, const Real c_1, const Real c_0,
        mptr<Real> x_out,
        const Real tol = 64 * Scalar::eps<Real>, const int max_iter = 64
    )
    {
        using Sign_T = FastInt8;
        
        (void)max_iter;
        
        constexpr Real a = 0;
        constexpr Real b = 1;
        
        constexpr Real zero = 0;
        
        // Computes the real roots of the cubic equation c_3 * x^3 + c_2 * x^2 + c_1 * x + c_0 = 0
        // in the interval [a,b]. We assume a < b.
        
        const Real TOL = tol * Abs(b-a);
        
        static_assert( Scalar::RealQ<Real>, "Input parameters must be of real floating point types.");
                
        constexpr Real infty = std::numeric_limits<Real>::max();
        
        auto f = [c_3,c_2,c_1,c_0]( const Real x )
        {
            return ((c_3 * x + c_2) * x + c_1) * x + c_0;
        };
        
        const Real Dc_3     = Scalar::Three<Real> * c_3;
        const Real Dc_3_inv = Inv( Dc_3 );
        
        const Real x_I = - c_2 * Dc_3_inv;
        const Real D   = c_2 * c_2 - c_1 * Dc_3;
        
        auto Solve = [f,TOL]( const Real L, const Real R, const Real x_init )
        {
            (void)x_init;
            return RegulaFalsi<0>( f, L, R, Scalar::Zero<Real>, TOL );
        };
     
        Real   cand [5];
        Sign_T sign [5];
        int    size = 0;
        
        cand[size] = a;
        sign[size] = Sign<Sign_T>(c_0);
        ++size;
        
        if( D > zero )
        {
            const Real delta = Abs( Sqrt(D) * Dc_3_inv );
            
            // The two critical points.
            const Real x_L = x_I - delta;
            const Real x_R = x_I + delta;
            
            if( a < x_L && x_L < b )
            {
                cand[size] = x_L;
                sign[size] = Sign<Sign_T>(f(x_L));
                ++size;
            }
            
            if( a < x_I && x_I < b )
            {
                cand[size] = x_I;
                sign[size] = Sign<Sign_T>(f(x_I));
                ++size;
            }
            
            if( a < x_R && x_R < b )
            {
                cand[size] = x_R;
                sign[size] = Sign<Sign_T>(f(x_R));
                ++size;
            }
            
        }
        else if( D < zero )
        {
            // At most one solution, as f is strictly increasing.
        }
        else if( D == zero )
        {
            // At most one solution, but x_I is strictly monotone.
            
            if( a < x_I && x_I < b )
            {
                cand[size] = x_I;
                sign[size] = Sign<Sign_T>(f(x_I));
                ++size;
            }
        }
        
        cand[size] = b;
        sign[size] = Sign<Sign_T>(c_0+c_1+c_2+c_3);
        ++size;
        
        int counter = 0;
        
        if( sign[0] == 0 )
        {
            x_out[counter++] = cand[0];
        }
        
        for( int i = 1; i < size; ++i )
        {
            if( OppositeSignQ( sign[i-1], sign[i] ) )
            {
                x_out[counter++] = Solve( cand[i-1], cand[i], Mean(cand[i-1], cand[i]) );
            }
            else if( sign[i] == 0 )
            {
                x_out[counter++] = cand[i];
            }
        }
        
        for( int i = counter; i < 3; ++i )
        {
            x_out[i] = infty;
        }
            
        return counter;
    }
    
    template<typename Real>
    int RealCubicSolve_UnitInterval_Newton(
        const Real c_3, const Real c_2, const Real c_1, const Real c_0,
        mptr<Real> x_out,
        const Real tol = 64 * Scalar::eps<Real>, const int max_iter = 64
    )
    {
        using Sign_T = FastInt8;
        
        constexpr Real a = 0;
        constexpr Real b = 1;
        
        constexpr Real zero = 0;
        
        // Computes the real roots of the cubic equation c_3 * x^3 + c_2 * x^2 + c_1 * x + c_0 = 0
        // in the interval [a,b]. We assume a < b.
        
        const Real TOL = tol * Abs(b-a);
        
        static_assert( Scalar::RealQ<Real>, "Input parameters must be of real floating point types.");
                
        constexpr Real infty = std::numeric_limits<Real>::max();
        
        auto f = [c_3,c_2,c_1,c_0]( const Real x )
        {
            return ((c_3 * x + c_2) * x + c_1) * x + c_0;
        };
        
        const Real Dc_3  = Scalar::Three<Real> * c_3;
        const Real Dc_2  = Scalar::Two<Real>   * c_2;
        
        const Real Dc_3_inv = Inv( Dc_3 );
        
        auto Df = [Dc_3,Dc_2,c_1]( const Real x )
        {
            return (( Dc_3 * x + Dc_2) * x + c_1);
        };
        
        const Real x_I = - c_2 * Dc_3_inv;
        const Real D   = c_2 * c_2 - Dc_3 * c_1;
        
        auto Solve = [=]( const Real L, const Real R, const Real x_init )
        {
            (void)L;
            (void)R;
            
            auto Newton = [=]( const Real x )
            {
                return  Max(a,Min(b, x - f(x)/Df(x)));
            };

            Real x_old = x_init;
            Real x     = Newton(x_init);

            int iter = 1;

            while( Abs(x-x_old) > TOL && iter < max_iter)
            {
                ++iter;
                x_old = x;
                x = Newton(x);
            }

            return x;
        };
        
        Real   cand [5];
        Sign_T sign [5];
        int    size = 0;
        
        cand[size] = a;
        sign[size] = Sign<Sign_T>(c_0);
        ++size;
        
        if( D > zero )
        {
            const Real delta = Sqrt(D) * Dc_3_inv;
            
            // The two critical points.
            const Real x_L = x_I - delta;
            const Real x_R = x_I + delta;
            
            if( a < x_L && x_L < b )
            {
                cand[size] = x_L;
                sign[size] = Sign<Sign_T>(f(x_L));
                ++size;
            }
            
            if( a < x_I && x_I < b )
            {
                cand[size] = x_I;
                sign[size] = Sign<Sign_T>(f(x_I));
                ++size;
            }
            
            if( a < x_R && x_R < b )
            {
                cand[size] = x_R;
                sign[size] = Sign<Sign_T>(f(x_R));
                ++size;
            }
            
        }
        else if( D < zero )
        {
            // At most one solution, as f is strictly increasing.
        }
        else if( D == zero )
        {
            // At most one solution, but x_I is strictly monotone.
            
            if( a < x_I && x_I < b )
            {
                cand[size] = x_I;
                sign[size] = Sign<Sign_T>(f(x_I));
                ++size;
            }
        }
        
        cand[size] = b;
        sign[size] = Sign<Sign_T>(c_0+c_1+c_2+c_3);
        ++size;
        
        int counter = 0;
        
        if( sign[0] == 0 )
        {
            x_out[counter++] = cand[0];
        }
        
        for( int i = 1; i < size; ++i )
        {
            if( OppositeSignQ( sign[i-1], sign[i] ) )
            {
                x_out[counter++] = Solve( cand[i-1], cand[i], Mean(cand[i-1], cand[i]) );
            }
            else if( sign[i] == 0 )
            {
                x_out[counter++] = cand[i];
            }
        }
        
        for( int i = counter; i < 3; ++i )
        {
            x_out[i] = infty;
        }
            
        return counter;
    }
    
    
    
    
    
    // From Graphics Gems
    
    template<typename Real, bool sortQ = true>
    int RealCubicSolve_Cardano( const Real b, const Real c, const Real d, mptr<Real> x )
    {
        // Computes the real roots of the cubic equation x^3 + b * x^2 + c * x + d = 0.

        static_assert( Scalar::RealQ<Real>, "Input parameters must be of real floating point type.");

        constexpr Real zero     = 0;
        constexpr Real two      = 2;
//        constexpr Real half  = Frac<Real>(1,2);
        constexpr Real third    = Frac<Real>(1,3);
        
        constexpr Real infty = std::numeric_limits<Real>::max();

        // Convert to depressed form:
        // t = x - shift
        // g(t) := f(t+shift) = t^3 - 2 * p * t + 2 * q

        const Real bb    = b * b;

        const Real p     = (c - bb * third) * third;

        const Real shift = - b * third;

        const Real q     = ((Frac<Real>(2,27) * bb - c * third) * b + d ) * Frac<Real>(1,2);

        const Real ppp   = p * p * p;
        
        const Real D     = q * q + ppp;


        if( D > zero )
        {
            // One real root.
            
            const Real sqrt_D = std::sqrt(D);
            
            const Real u = std::cbrt( - q + sqrt_D );
            const Real v = std::cbrt( - q - sqrt_D );;

            x[0] = u + v + shift;
            x[1] = infty;
            x[2] = infty;
            
            return 1;
        }
        else if ( D < zero )
        {
            // Three real roots. (casus irreducibilis)
            
            const Real theta =  third * std::acos( - q / std::sqrt( -ppp ) );

//            constexpr Real pi_third = Scalar::Pi<Real> * third;
//            
//            const Real two_sqrt_minus_p = two * std::sqrt( - p );
//            
//            x[0] = shift + two_sqrt_minus_p * std::cos( theta            );
//            x[1] = shift - two_sqrt_minus_p * std::cos( theta + pi_third );
//            x[2] = shift - two_sqrt_minus_p * std::cos( theta - pi_third );
            
            const Real sqrt_minus_p = std::sqrt( - p );
            
            const Real A = std::cos( theta );
            const Real B = cSqrt(Scalar::Three<Real>) * std::sin( theta );
            
            x[0] = shift + two * sqrt_minus_p * (A    );
            x[1] = shift -       sqrt_minus_p * (A + B);
            x[2] = shift -       sqrt_minus_p * (A - B);
            
            if constexpr ( sortQ )
            {
                SortNet<3> S;
                
                S(&x[0]);
            }

            return 3;
        }
        else
        {
            // One double root.
            
            x[0] = x[1] = shift + std::cbrt( - q );
            x[2] = infty;
            
            return 2;
        }
    } // RealCubicSolve_Cardano
    
    
    
    
    template<typename Real, bool sortQ = true>
    int RealCubicSolve_Cardano(
        const Real a, const Real b, const Real c, const Real d, mptr<Real> x
    )
    {
        // Computes the real roots of the cubic equation a * x^3 + b * x^2 + c * x + d = 0.
        
        static_assert( Scalar::RealQ<Real>, "Input parameters must be of real floating point types.");
        
        constexpr Real half = Inv<Real>(2);
        
        constexpr Real eps   = 8 * std::numeric_limits<Real>::epsilon();
        constexpr Real infty = std::numeric_limits<Real>::max();
        
        if( std::abs(a) < eps )
        {
            if( std::abs(b) < eps )
            {
                if( std::abs(c) < eps )
                {
//                    print("constant equation");
                    
                    if( std::abs(d) < eps )
                    {
                        x[0] = Scalar::Zero<Real>;
                        x[1] = infty;
                        x[2] = infty;
                        
                        return 1;
                    }
                    else
                    {
                        x[0] = infty;
                        x[1] = infty;
                        x[2] = infty;
                        
                        return 0;
                    }
                }
                else
                {
//                    print("linear equation");
                    
                    x[0] = -d/c;
                    x[1] = infty;
                    x[2] = infty;
                    
                    return 1;
                }
            }
            else
            {
//                print("quadratic equation");
                
                const Real p = c/b;
                const Real q = d/b;
                const Real discriminant = p * p - Scalar::Four<Real> * q;
                
                if( std::abs(discriminant) < eps )
                {
//                    print("no real solution");
                    
                    x[0] = infty;
                    x[1] = infty;
                    x[2] = infty;
                    
                    return 0;
                }
                else
                {
//                    print("two real solutions");
                    const Real sqrt_discriminant = Sqrt(discriminant);
                    x[0] = half * ( -p - sqrt_discriminant );
                    x[1] = half * ( -p + sqrt_discriminant );
                    x[2] = infty;
                    
                    return 2;
                }
            }
        }
        else
        {
//            print("cubic equation");
            
            const Real a_inv = Inv<Real>(a);
            return RealCubicSolve_Cardano<Real,sortQ>( b * a_inv, c * a_inv, d * a_inv, x );
        }
        
    } // RealCubicSolve_Cardano
    
    
    template<typename Real>
    int RealCubicSolve_UnitInterval_Cardano(
        const Real a, const Real b, const Real c, const Real d, mptr<Real> x
    )
    {
        int count = RealCubicSolve_Cardano<Real,false>( a, b, c, d, x );
        
        for( int i = count; i --> int(0); )
        {
            if( (x[i] < int(0)) || ( x[i] > int(1) ) )
            {
                --count;
                x[i] = Scalar::Max<Real>;
            }
        }
        
        SortNet<3> S;
        
        S(&x[0]);
        
        return count;
        
        
    } // RealCubicSolve_UnitInterval_Cardano
    
    
} // namespace Tools

 
