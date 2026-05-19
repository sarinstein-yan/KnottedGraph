#pragma once

namespace Tools
{
    namespace Scalar
    {
        
//###########################################################
//##  Complex-Real
//###########################################################
        
        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && OpReturnRealQ<a_flag,op,R,Cplx<R>>(),
            R
        >
        Op( cref<Cplx<R>> a, cref<R> x )
        {
            return Op_vec<a_flag,op>(a,x);
        }
        
        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && !OpReturnRealQ<a_flag,op,R,Cplx<R>>(),
            Cplx<R>
        >
        Op( cref<Cplx<R>> a, cref<R> x )
        {
            return to_Cplx(Op_vec<a_flag,op>(a,x));
        }
    
//###########################################################
//##  Real-Complex
//###########################################################
    
        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && OpReturnRealQ<a_flag,op,R,Cplx<R>>(),
            R
        >
        Op( cref<R> a, cref<Cplx<R>> x )
        {
            return Op_vec<a_flag,op>(a,x);
        }
        
        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && !OpReturnRealQ<a_flag,op,R,Cplx<R>>(),
            Cplx<R>
        >
        Op( cref<R> a, cref<Cplx<R>> x )
        {
            return to_Cplx(Op_vec<a_flag,op>(a,x));
        }
    
    
//###########################################################
//##  Complex-Complex
//###########################################################

        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && OpReturnRealQ<a_flag,op,Cplx<R>,Cplx<R>>(),
            R
        >
        Op( cref<Cplx<R>> a, cref<Cplx<R>> x )
        {
            return Op_vec<a_flag,op>(a,x);
        }
        
        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && !OpReturnRealQ<a_flag,op,Cplx<R>,Cplx<R>>(),
            Cplx<R>
        >
        Op( cref<Cplx<R>> a, cref<Cplx<R>> x )
        {
            return to_Cplx(Op_vec<a_flag,op>(a,x));
        }

    } // namespace Scalar
        
} // namespace Tools
