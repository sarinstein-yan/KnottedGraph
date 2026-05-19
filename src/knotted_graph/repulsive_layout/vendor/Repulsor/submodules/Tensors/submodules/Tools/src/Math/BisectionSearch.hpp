#pragma once

namespace Tools
{
    
    template<int up_lo_flag, typename Real, typename Fun_T>
    TOOLS_FORCE_INLINE Real BisectionSearch(
        const Fun_T & f, const Real a_0, const Real b_0, const Real value,
        const Real relative_error
    )
    {
        // If up_lo_flag == 0, then return interval midpoint.
        // If up_lo_flag >  0, then return interval endpoint with higher function value.
        // If up_lo_flag <  0, then return interval endpoint with lower function value.
        
        Real a = Min(a_0,b_0);
        Real b = Max(a_0,b_0);
        
        Real f_a = f(a);
        Real f_b = f(b);
        
        if( (f_a <= value) && (value < f_b) )
        {
            // f_a <= v < f_b;
            
            while( b - a > relative_error * b )
            {
                const Real c = Mean(a,b);
                
                const Real f_c = f(c);
                
                if( f_c > value )
                {
                    b   = c;
                    f_b = f_c;
                }
                else
                {
                    a   = c;
                    f_a = f_c;
                }
            }
            
            if constexpr ( up_lo_flag == int(0) )
            {
                return Mean(a,b);
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
        else if( (f_b <= value) && (value < f_a) )
        {
            // f_b <= v < f_a;
            
            while( b - a > relative_error * b )
            {
                const Real c = Mean(a,b);
                
                const Real f_c = f(c);
                
                if( f_c > value )
                {
                    a   = c;
                    f_a = f_c;
                }
                else
                {
                    b   = c;
                    f_b = f_c;
                }
            }
                  
            if constexpr ( up_lo_flag == int(0) )
            {
                return Mean(a,b);
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
            wprint("BisectionSearch: The functions values of the interval end points do not bracket the value f. Returning interval midpoint.");
                
            return Mean(a,b);
        }
        
    } // BisectionSearch
    
    
}

 

