#pragma once

namespace Tools
{
    
    template<typename Real>
    TOOLS_FORCE_INLINE std::pair<Real,Real> TwoSum( const Real a, const Real b )
    {
        // Algorithm 3.1 from Ogita, Rump, Oishi - Accurate Sum and Dot Product
        //
        // Compute a + b in terms of x and y: x + y = a + b;
        // |y| << |x|.
        
        // Trying to turn off floating point optimizations like those allowed by the compiler flags -fassociative-math or -ffast-math
        TOOLS_MAKE_FP_STRICT()
        
        static_assert(std::is_floating_point_v<Real>,"");
        
        const Real x = a + b;
        const Real z = x - a;
        const Real u = x - z;
        const Real v = a - u;
        const Real w = b - z;
        const Real y = v + w;

        return {x,y};
    }
    
    
//    TOOLS_FORCE_INLINE std::pair<double,double> TwoSum_asm_arm_reg( double a, double b )
//    {
//        // In my version of clang this should be the precise equivalent to TwoSum without -ffast-math.
//        
//        double x,y,z;
//            
//        asm( R"(
//                fadd %d[x], %d[a], %d[b]
//                fsub %d[y], %d[x], %d[a]
//                fsub %d[z], %d[x], %d[y]
//                fsub %d[b], %d[b], %d[y]
//                fsub %d[a], %d[a], %d[z]
//                fadd %d[b], %d[b], %d[a]
//                fmov %d[a], %d[x]
//            )"
//            :
//                [a] "+w"  (a),
//                [b] "+w"  (b),
//                [x] "=&w" (x),
//                [y] "=&w" (y),
//                [z] "=&w" (z)
//            :
//        );
//
//        return {a,b};
//    }
//    
//    TOOLS_FORCE_INLINE std::pair<double,double> TwoSum_asm_arm( const double a, const double b )
//    {
//        // My poor man's transformation of TwoSum to asm.
//        
//        // Surprisingly, it is _faster_ than what clang procudes in TwoSum_asm_arm_reg, albeit using more registers.
//        
//        // Maybe the dependency chain is shorter so that some few things can be computed concurrently?
//        
//        // TODO: There is one superfluous register move here. Can we get rid of it?
//        
//        double u,v,w,x,y,z;
//
//        asm( R"(
//
//            fadd %d[x], %d[a], %d[b] // x = a + b; 
//                                                 ; --+-+      
//                                                 ;   | |
//                                                 ; <-+ |
//            fsub %d[z], %d[x], %d[a] // z = x - a; ----|--+
//                                                 ; --+ |  |
//                                                 ;   | |  |
//                                                 ; <-+ |  |
//            fsub %d[u], %d[x], %d[z] // u = x - z; <---+  |
//                                                 ; --+    |
//                                                 ;   |    |
//                                                 ; <-+    |
//            fsub %d[v], %d[a], %d[u] // v = a - u;        |
//                                                 ; ----+  |
//                                                 ;     |  |
//                                                 ; <-+-|--+
//            fsub %d[w], %d[b], %d[z] // w = b - z;     |  
//                                                 ; --+ |
//                                                 ;   | |
//                                                 ; <-+ |
//            fadd %d[y], %d[v], %d[w] // y = v + w; <---+
//            )"
//            :
//                [u] "=&w" (u),
//                [v] "=&w" (v),
//                [w] "=&w" (w),
//                [x] "=&w" (x),
//                [y] "=&w" (y),
//                [z] "=&w" (z)
//            :
//                [a] "w" (a),
//                [b] "w" (b)
//        );
//
//        return {x,y};
//    }
//    
//
//    TOOLS_FORCE_INLINE std::pair<float,float> TwoSum_asm_arm( const float a, const float b )
//    {
//        // My poor man's transformation of TwoSum to asm.
//        
//        // Surprisingly, it is _faster_ than what clang procudes in TwoSum_asm_arm_reg, albeit using more registers.
//        
//        // Maybe the dependency chain is shorter so that some few things can be computed concurrently?
//        
//        // TODO: There is one superfluous register move here. Can we get rid of it?
//        
//        float u,v,w,x,y,z;
//
//        asm( R"(
//
//            fadd %s[x], %s[a], %s[b] // x = a + b; 
//                                                 ; --+-+      
//                                                 ;   | |
//                                                 ; <-+ |
//            fsub %s[z], %s[x], %s[a] // z = x - a; ----|--+
//                                                 ; --+ |  |
//                                                 ;   | |  |
//                                                 ; <-+ |  |
//            fsub %s[u], %s[x], %s[z] // u = x - z; <---+  |
//                                                 ; --+    |
//                                                 ;   |    |
//                                                 ; <-+    |
//            fsub %s[v], %s[a], %s[u] // v = a - u;        |
//                                                 ; ----+  |
//                                                 ;     |  |
//                                                 ; <-+-|--+
//            fsub %s[w], %s[b], %s[z] // w = b - z;     |  
//                                                 ; --+ |
//                                                 ;   | |
//                                                 ; <-+ |
//            fadd %s[y], %s[v], %s[w] // y = v + w; <---+
//            )"
//            :
//                [u] "=&w" (u),
//                [v] "=&w" (v),
//                [w] "=&w" (w),
//                [x] "=&w" (x),
//                [y] "=&w" (y),
//                [z] "=&w" (z)
//            :
//                [a] "w" (a),
//                [b] "w" (b)
//        );
//
//        return {x,y};
//    }
    
    
    
    template<typename Real>
    TOOLS_FORCE_INLINE std::pair<Real,Real> TwoProductFMA( const Real a, const Real b )
    {
        // Algorithm 3.5 from Ogita, Rump, Oishi - Accurate Sum and Dot Product
        //
        // Compute a * b in terms of x and y: x + y = a * b;
        // |y| << |x|.
        
        // std::fma should make this work even with -ffast-math compiler flag. Just to be safe:
        TOOLS_MAKE_FP_STRICT()
        
        static_assert(std::is_floating_point_v<Real>,"");
        
        const Real x = a * b;
        const Real y = std::fma(a, b, -x);
        
        return {x,y};
    }
    
    
    template<typename Real_>
    class CompensatedAccumulator
    {
    public:
        
        using Real = Real_;

    private:
        
        Real sum_hi = 0;
        Real sum_lo = 0;
        
    public:
        
        CompensatedAccumulator()
        :   sum_hi{ 0 }
        {}
        
        CompensatedAccumulator( const Real init )
        :   sum_hi{ init }
        {}
        
        ~CompensatedAccumulator() = default;
        
        CompensatedAccumulator & operator+=( const Real x )
        {
            // Algorithm 4.3 "Sum2" from Ogita, Rump, Oishi - Accurate Sum and Dot Product
            
            Real r;
            
            std::tie( sum_hi, r ) = TwoSum( x, sum_hi );
            
//            std::tie( sum_hi, r ) = TwoSum_asm_arm( x, sum_hi );
            
//            std::tie( sum_hi, r ) = TwoSum_asm_arm_reg( x, sum_hi );
            
            sum_lo += r;
            
            return *this;
        }
        
        Real Value() const
        {
            return sum_hi + sum_lo;
        }
        
        std::pair<Real,Real> Values() const
        {
            return {sum_hi, sum_lo};
        }
        
    }; // CompensatedAccumulator
    
    
    // TODO: Make it parallel!
    template<Size_T N = 0, typename Real>
    TOOLS_FORCE_INLINE Real CompensatedSum( cptr<Real> x, const Size_T n = N )
    {
        CompensatedAccumulator<Real> acc( x[0] );
        
//        TOOLS_DUMP(acc.Value());
        
        for( Size_T i = 1; i < n; ++i )
        {
            acc += x[i];
            
//            TOOLS_DUMP(acc.Value());
        }
        
        return acc.Value();
    }

    template<typename Real>
    TOOLS_FORCE_INLINE Real CompensatedSum( const Real a, const Real b )
    {
        CompensatedAccumulator<Real> acc( a );

        acc += b;
        
        return acc.Value();
    }

    template<typename Real>
    TOOLS_FORCE_INLINE Real CompensatedSum( const Real a, const Real b, const Real c )
    {
        CompensatedAccumulator<Real> acc( a );

        acc += b;
        acc += c;
        
        return acc.Value();
    }
    
    
    template<Size_T N = 0, typename Real>
    TOOLS_FORCE_INLINE Real CompensatedDot( cptr<Real> x, cptr<Real> y, const Size_T n = N )
    {
        // Algorithm 5.3 "Dot2" from Ogita, Rump, Oishi - Accurate Sum and Dot Product

        Real h;
        Real l;
        Real a;
        Real b;
        Real c;
                
        std::tie(h,l) = TwoProductFMA(x[0],y[0]);

        for( Size_T i = 1; i < n; ++i )
        {
            std::tie(a,b) = TwoProductFMA(x[i],y[i]);
            
            std::tie(h,c) = TwoSum(h,a);
//            std::tie(h,c) = TwoSum_asm_arm(h,a);

            l += (c+b);
        }

        return h + l;
    }
    
} // namespace Tools
