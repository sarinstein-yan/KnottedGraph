#pragma once

namespace Tools
{
    
    // Special version for purely complex types.
    template< enum Tools::Op op = Tools::Op::Id, typename R>
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<MatrixizableQ<R>,mat_T<2,2,R>>
    ComplexMultiplierMatrix( const Cplx<R> & a)
    {
        static_assert(NotTransposedQ(op),"");

        mat_T<2,2,R> A;

        if constexpr ( op == Tools::Op::Id )
        {
            A[0][0] =  Re(a);   A[0][1] = -Im(a);
            A[1][0] =  Im(a);   A[1][1] =  Re(a);
        }
        else if constexpr ( op == Tools::Op::Re )
        {
            A[0][0] =  Re(a);   A[0][1] =  0;
            A[1][0] =  Im(a);   A[1][1] =  0;
        }
        else if constexpr ( op == Tools::Op::Im )
        {
            A[0][0] =  0;       A[0][1] = -Im(a);
            A[1][0] =  0;       A[1][1] =  Re(a);
        }
        else if constexpr ( op == Tools::Op::Conj )
        {
            A[0][0] =  Re(a);   A[0][1] =  Im(a);
            A[1][0] =  Im(a);   A[1][1] = -Re(a);
        }

        return A;
    }

    // Special version for purely complex types.
    template< enum Tools::Op op = Tools::Op::Id, typename R>
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<MatrixizableQ<mat_T<2,2,R>>,void>
    ComplexMultiplierMatrixT( const Cplx<R> & a)
    {
        static_assert(NotTransposedQ(op),"");

        mat_T<2,2,R> A;

        if constexpr ( op == Tools::Op::Id )
        {
            A[0][0] =  Re(a);   A[0][1] =  Im(a);
            A[1][0] = -Im(a);   A[1][1] =  Re(a);
        }
        else if constexpr ( op == Tools::Op::Re )
        {
            A[0][0] =  Re(a);   A[0][1] =  Im(a);
            A[1][0] =  0;       A[1][1] =  0;
        }
        else if constexpr ( op == Tools::Op::Im )
        {
            A[0][0] =  0;       A[0][1] =  0;
            A[1][0] = -Im(a);   A[1][1] =  Re(a);
        }
        else if constexpr ( op == Tools::Op::Conj )
        {
            A[0][0] =  Re(a);   A[0][1] =  Im(a);
            A[1][0] =  Im(a);   A[1][1] = -Re(a);
        }

        return A;
    }
    

    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename Real
    >
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<MatrixizableQ<Real>,void>
    combine_scalars_1(
        cref<Cplx<Real>> a, cref<Cplx<Real>> x,
        // We explicitly allopw y and z to overlap.
        cref<Cplx<Real>> b, const Cplx<Real> & y,
        Cplx<Real> & z
    )
    {
        // This computes z = a * opx(x) + b * opy(y).
        
        // We use references for the arguments in the case that
        // this would require to dereference some illegal pointer
        // upstream.
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
        using V_T = mat_T<2,1,Real>;
        
        // to_vec_T is safe because x, y, z are guaranteed to be complex.
        
        const V_T & X = to_vec_T(x);
        const V_T & Y = to_vec_T(y);
              V_T & Z = to_vec_T(z);
        
        // WARNING: Currently, this always uses generic case.
        // TODO: Improve implementation for opx == Op::Re, opx == Op::Im, opy == Op::Re, opy == Op::Im
        
        {
            TOOLS_MAKE_FP_FAST()
        
            Z = ComplexMultiplierMatrix<opx>(a) * X
                +
                ComplexMultiplierMatrix<opy>(b) * Y;
        }
    }
    
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename Real
    >
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<MatrixizableQ<Real>,void>
    combine_scalars_1T(
        cref<Cplx<Real>> a, cref<Cplx<Real>> x,
        // We explicitly allopw y and z to overlap.
        cref<Cplx<Real>> b, const Cplx<Real> & y,
        Cplx<Real> & z
    )
    {
        // This computes z = a * opx(x) + b * opy(y).
        
        // We use references for the arguments in the case that
        // this would require to dereference some illegal pointer
        // upstream.
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
        using V_T = mat_T<1,2,Real>;
        
        // to_vec_T is safe because x, y, z are guaranteed to be complex.
        
        const V_T & X = to_vec_T(x);
        const V_T & Y = to_vec_T(y);
              V_T & Z = to_vec_T(z);
        
        // WARNING: Currently, this always uses generic case.
        // TODO: Improve implementation for opx == Op::Re, opx == Op::Im, opy == Op::Re, opy == Op::Im
        
        {
            TOOLS_MAKE_FP_FAST()
        
            Z = X * ComplexMultiplierMatrixT<opx>(a)
                +
                Y * ComplexMultiplierMatrixT<opy>(b);
        }
    }
    

    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename Real
    >
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<MatrixizableQ<Real>,void>
    combine_scalars_3(
        cref<Cplx<Real>> a, cref<Cplx<Real>> x,
        // We explicitly allow y and z to overlap.
        cref<Cplx<Real>> b, const Cplx<Real> & y,
        Cplx<Real> & z
    )
    {
        // This variant might work better for cases with SIMD length 4.
        // Or it might not.
        
        // This computes z = a * opx(x) + b * opy(y).
        
        // We use references for the arguments in the case that
        // this would require to dereference some illegal pointer
        // upstream.
        
        static_assert( NotTransposedQ(opx),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opx."
        );
        
        static_assert( NotTransposedQ(opy),
            "combine_scalars: Only those values of Op that do not inlude transposition are allowed for template argument opy."
        );
        
        using V_T = mat_T<2,1,Real>;
        using W_T = mat_T<4,1,Real>;
        using M_T = mat_T<2,4,Real>;
        
        {
            TOOLS_MAKE_FP_FAST()
            
            M_T AB;
            
            AB[0][0] =  Re(a);
            AB[0][1] = -Im(a);
            AB[0][2] =  Re(b);
            AB[0][3] = -Im(b);
            
            AB[1][0] =  Im(a);
            AB[1][1] =  Re(a);
            AB[1][2] =  Im(b);
            AB[1][3] =  Re(b);
            
            Real XY_buf [4] { Re(x), Im(x), Re(y), Im(y) };
            
            // TODO: This is unsafe.
            
            const W_T & XY = *reinterpret_cast<const W_T *>(&XY_buf[0]   );
            
                  V_T &  Z = get_vec_T(&z);
            
            Z = AB * XY;
        }
    }
    
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename Real
    >
    TOOLS_FORCE_INLINE constexpr std::enable_if_t<MatrixizableQ<Real>,void>
    combine_scalars_3_perm(
        cref<Cplx<Real>> a, cref<Cplx<Real>> x,
        // We explicitly allopw y and z to overlap.
        cref<Cplx<Real>> b, const Cplx<Real> & y,
        Cplx<Real> & z
    )
    {
        // This variant might work better for cases with SIMD length 4.
        // Or it might not.
        
        using V_T = mat_T<2,1,Real>;
        using W_T = mat_T<4,1,Real>;
        using M_T = mat_T<2,4,Real>;
        
        {
            TOOLS_MAKE_FP_FAST()
            
            M_T AB;
            
            AB[0][0] =  Re(a);
            AB[0][1] =  Re(b);
            AB[0][2] = -Im(a);
            AB[0][3] = -Im(b);
            
            AB[1][0] =  Im(a);
            AB[1][1] =  Im(b);
            AB[1][2] =  Re(a);
            AB[1][3] =  Re(b);
            
            Real XY_buf [4] { Re(x), Re(y), Im(x), Im(y) };
            
            // TODO: This is unsafe.
            
            const W_T & XY = *reinterpret_cast<const W_T *>(&XY_buf[0]   );
            
                  V_T &  Z = get_vec_T(&z);
            
            Z = AB * XY;
        }
    }
    

    // Special version for purely complex types.
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename Real
    >
    TOOLS_FORCE_INLINE constexpr
    std::enable_if_t<MatrixizableQ<Real>,void> combine_scalars(
        cref<Cplx<Real>> a, cref<Cplx<Real>> x,
        cref<Cplx<Real>> b, const Cplx<Real> & y,
        // We explicitly allow y and z to overlap.
        Cplx<Real> & z
    )
    {
        combine_scalars_1<a_flag,b_flag,opx,opy>(a,x,b,y,z);
//        combine_scalars_1T<a_flag,b_flag,opx,opy>(a,x,b,y,z);
//        combine_scalars_2<a_flag,b_flag,opx,opy>(a,x,b,y,z);
//        combine_scalars_3<a_flag,b_flag,opx,opy>(a,x,b,y,z);
//        combine_scalars_3_perm<a_flag,b_flag,opx,opy>(a,x,b,y,z);
    }
    
    
    // Faster versions for purely complex types.
    template<
        Scalar::Flag a_flag, Scalar::Flag b_flag,
        Op opx = Op::Id, Op opy = Op::Id,
        typename Real
    >
    TOOLS_FORCE_INLINE constexpr
    std::enable_if_t<MatrixizableQ<Real>,void> combine_scalars(
        cref<Cplx<Real>> a, cref<Cplx<Real>> x,
        cref<Cplx<Real>> b, mref<Cplx<Real>> y
    )
    {
        // This computes y = a * opx(x) + b * opy(y).

        combine_scalars<a_flag,b_flag,opx,opy>(a,x,b,y,y);
    }
    
    
    
} // namespace Tools
