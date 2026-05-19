#pragma once

namespace Tools
{
    
    template<int up_lo_flag, typename Real, typename Fun_T>
    inline Real RegulaFalsi(
        const Fun_T & f, const Real a_0, const Real b_0, const Real value,
        const Real relative_error
    )
    {
        // "Illinois algorithm" from
        // https://en.wikipedia.org/wiki/Regula_falsi#Example_code
        
        // If up_lo_flag == 0, then return interval midpoint.
        // If up_lo_flag >  0, then return interval endpoint with higher function value.
        // If up_lo_flag <  0, then return interval endpoint with lower function value.
        
        constexpr Real zero = 0;
        
        const Real e = Scalar::Half<Real> * relative_error;
        
        Real a = Min(a_0,b_0);
        Real b = Max(a_0,b_0);
        
        Real f_a = f(a) - value;
        Real f_b = f(b) - value;
        
        Real c     = b;
        Real f_c   = f_b;
        
        int side = 0;
        
        if( f_a == zero )
        {
            return a;
        }
        else if ( f_b == 0 )
        {
            return b;
        }
        
        if( (f_a < zero) && (zero < f_b) )
        {
            // f_a <= v < f_b;
            
            while( true )
            {
                
                if( Abs(b - a) < e * Abs(b + a) )
                {
                    break;
                }
                
                c = (f_a * b - f_b * a) / (f_a - f_b);
                
                f_c = f(c) - value;
                
                if( f_c > zero )
                {
                    b   = c;
                    f_b = f_c;
                    if( side == -1 )
                    {
                        f_a *= Scalar::Half<Real>;
                    }
                    side = -1;
                }
                else
                {
                    a   = c;
                    f_a = f_c;
                    if( side == 1 )
                    {
                        f_b *= Scalar::Half<Real>;
                    }
                    side = 1;
                }
            }
            
            if constexpr ( up_lo_flag == int(0) )
            {
                return (f_a * b - f_b * a) / (f_a - f_b);
            }
            else if constexpr ( up_lo_flag < int(0) )
            {
                return a;
            }
            else if constexpr ( up_lo_flag > int(0) )
            {
                return b;
            }
        }
        else if( (f_b < zero) && (zero < f_a) )
        {
            // f_b <= v < f_a;
            
            while( true )
            {
                if( Abs(b - a) < e * Abs(b + a) )
                {
                    break;
                }

                c = ( b * f_a - a * f_b ) / ( f_a - f_b );
                
                f_c = f(c) - value;
                
                if( f_c > zero )
                {
                    a   = c;
                    f_a = f_c;
                    if( side == int(1) )
                    {
                        f_b *= Scalar::Half<Real>;
                    }
                    side = 1;
                }
                else
                {
                    b   = c;
                    f_b = f_c;
                    if( side == int(-1) )
                    {
                        f_a *= Scalar::Half<Real>;
                    }
                    side = -1;
                }
            }
            
            if constexpr ( up_lo_flag == int(0) )
            {
                return ( b * f_a - a * f_b ) / ( f_a - f_b );
            }

            else if constexpr ( up_lo_flag < int(0) )
            {
                return b;
            }
            else if constexpr ( up_lo_flag > int(0) )
            {
                return a;
            }
        }
        else
        {
            wprint("RegularFalsi: The functions values of the interval end points do not bracket the value f. Returning interval midpoint.");
            
            TOOLS_DUMP(f_a);
            TOOLS_DUMP(f_b);
            TOOLS_DUMP(Sign(f_a));
            TOOLS_DUMP(Sign(f_b));
                
            return Mean(a,b);
        }
        
    } // RegularFalsi
    
    
}

 

