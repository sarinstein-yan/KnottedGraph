#pragma once

namespace Tools
{

    struct OuterProd_TArg
    {
        Scalar::Flag a_flag = Scalar::Flag::Generic;
        Op opx = Op::Id; Size_T nx = VarSize;
        Op opy = Op::Id; Size_T ny = VarSize;
        Scalar::Flag b_flag = Scalar::Flag::Generic;
        Parallel_T parQ = Sequential;
    };
    
    template<OuterProd_TArg t>
    struct OuterProd
    {
        template <
            typename x_T, typename y_T, typename C_T,
            typename a_T = Scalar::Real<x_T>,
            typename b_T = Scalar::Real<C_T>,
            IntQ     Int = Size_T
        >
        struct Kernel
        {
            static_assert( NotTransposedQ(t.opx), "" );
            static_assert( NotTransposedQ(t.opy), "" );
            
            using F_T = Scalar::Flag;
         
            // We initialize this to 1 if we have no reason to do otherwise.
            const a_T a   = (t.a_flag == F_T::Minus ? a_T(-1) :
                              (t.a_flag == F_T::Zero ? a_T(0) : a_T(1) ) );
            cptr<x_T> x;
            const Int nx  = t.nx;
            const Int ldx = nx;
            cptr<y_T> y;
            const Int ny  = t.ny;
//            const Int ldy = t.ny;
            // We initialize this to 0 if we have no reason to do otherwise.
            const b_T b   = (t.b_flag == F_T::Minus ? b_T(-1) :
                              (t.b_flag == F_T::Plus ? b_T(1) : b_T(0) ) );
            mptr<C_T> C;
            const Int ldC = nx * ny;
            const Int thread_count = 1;
            
            void operator()()
            {
                constexpr Scalar::Flag x_flag =
                    (t.a_flag == Scalar::Flag::Zero)
                    ? Scalar::Flag::Zero
                    : Scalar::Flag::Generic;
                

                Do<t.nx,t.parQ>(
                    [this](const Int i )
                    {
                        const auto ax_i = Scalar::Op<t.a_flag,t.opx>(a,x[ldx * i]);
                        
                        combine_buffers<x_flag,t.b_flag,t.ny,Sequential,t.opy>(
                            ax_i, y,
                            b, &C[ldC * i], ny
                        );
                    },
                    nx, thread_count
                );
                
            } // operator()
            
        }; // Kernel
        
    }; // OuterProd
        
} // namespace Tools
