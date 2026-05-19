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
        Op_vec( cref<Cplx<R>> a, cref<R> x )
        {
            // It vectorizable, then real, right?
            static_assert(RealQ<R>,"");
            
            if constexpr ( (a_flag == Flag::Zero) || (op == Tools::Op::Im) || (op == Tools::Op::Im) )
            {
                return 0;
            }
            else if constexpr ( (a_flag == Flag::Plus) )
            {
                return x;
            }
            else if constexpr ( (a_flag == Flag::Minus) )
            {
                return -x;
            }
            else // if constexpr ( (a_flag == Flag::Generic) )
            {
                // If we land here, then `a` is complex and `x` is real.
                // Moreover, `a` is neither `0`, `-1`, or `-1` (or rather: we cannot tell this at compile time).
                // And we cannot tell whether `op(x)` is zero.
                
                (void)a;
                
                static_assert(Tools::DependentFalse<R>,"");
            }
        }
        
        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && !OpReturnRealQ<a_flag,op,R,Cplx<R>>(),
            vec_T<2,R>
        >
        Op_vec( cref<Cplx<R>> a, cref<R> x )
        {
            // It vectorizable, then real, right?
            static_assert(RealQ<R>,"");
            
            using V_T = vec_T<2,R>;
            
            if constexpr ( a_flag == Flag::Generic )
            {
                V_T & A = *reinterpret_cast<const V_T *>(&a);
                return A * x;
            }
            else // if constexpr ( a_flag != Flag::Generic )
            {
                // If we land here, then `x` is real and `a` must be either `0`, `1`, or `-1`. So, OpReturnRealQ should have returned `true`.
                static_assert(Tools::DependentFalse<R>,"");
            }
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
        Op_vec( cref<R> a, cref<Cplx<R>> x )
        {
            if constexpr ( a_flag == Flag::Zero )
            {
                return 0;
            }
            else if constexpr ( (op == Tools::Op::Re) || (op == Tools::Op::ReTrans) )
            {
                return Op<a_flag,Tools::Op::Id>(a,Re(x));
            }
            else if constexpr ( (op == Tools::Op::Im) || (op == Tools::Op::ImTrans) )
            {
                return Op<a_flag,Tools::Op::Id>(a,Im(x));
            }
            else
            {
                // If we land here, then we cannot certify at compile time that op(x) is real and `a != 0`. So we cannot guarantee that `a * op(x)` is real -- and OpReturnRealQ should have returned false. So this should be impossible.
                static_assert(Tools::DependentFalse<R>,"");
            }
        }

        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && !OpReturnRealQ<a_flag,op,R,Cplx<R>>(),
            vec_T<2,R>
        >
        Op_vec( cref<R> a, cref<Cplx<R>> x )
        {
            using V_T = vec_T<2,R>;
            
            const V_T & X = *reinterpret_cast<const V_T * >(&x);
            
            if constexpr ( a_flag == Flag::Plus )
            {
                if constexpr ( op == Tools::Op::Id )
                {
                    return X;
                }
                else if constexpr ( op == Tools::Op::Conj )
                {
                    return V_T{Re(x),-Im(x)};
                }
                else
                {
                    // If we land here, then op(x) is guaranteed to be real.
                    // And since `a` is real as well, `OpReturnRealQ` should have returned true. So this is impossible.
                    static_assert(Tools::DependentFalse<R>,"");
                }
            }
            else if constexpr ( a_flag == Flag::Minus )
            {
                if constexpr ( op == Tools::Op::Id )
                {
                    return -X;
                }
                else if constexpr ( op == Tools::Op::Conj )
                {
                    return V_T{-Re(x),Im(x)};
                }
                else
                {
                    // If we land here, then op(x) is guaranteed to be real.
                    // And since `a` is real as well, `OpReturnRealQ` should have returned true. So this is impossible.
                    static_assert(Tools::DependentFalse<R>,"");
                }
            }
            else if constexpr ( a_flag == Flag::Generic )
            {
                if constexpr ( op == Tools::Op::Id )
                {
                    V_T A_0 {  Re(a),  Im(a) };
                    V_T A_1 { -Im(a),  Re(a) };
                    
                    return A_0 * Re(x) + A_1 * Im(x);
                }
                else if constexpr ( op == Tools::Op::Conj )
                {
                    V_T A_0 {  Re(a),  Im(a) };
                    V_T A_1 {  Im(a), -Re(a) };
                    
                    return A_0 * Re(x) + A_1 * Im(x);
                }
                else
                {
                    // If we land here, then op(x) is guaranteed to be real.
                    // And since `a` is real as well, `OpReturnRealQ` should have returned true. So this is impossible.
                    
                    static_assert(Tools::DependentFalse<R>,"");
                }
            }
            else // if constexpr ( a_flag == Flag::Zero )
            {
                // If we land here, then `a == 0`, so `a * op(x) == 0` and `OpReturnRealQ` should have returned true. So this is impossible.
                static_assert(Tools::DependentFalse<R>,"");
            }
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
        Op_vec( cref<Cplx<R>> a, cref<Cplx<R>> x )
        {
            if constexpr ( (a_flag == Flag::Zero) )
            {
                return 0;
            }
            else if constexpr ( (op == Tools::Op::Re) || (op == Tools::Op::ReTrans) )
            {
                return Op_vec<a_flag,Tools::Op::Id>(a,Re(x));
            }
            else if constexpr ( (op == Tools::Op::Im) || (op == Tools::Op::ImTrans) )
            {
                return Op_vec<a_flag,Tools::Op::Id>(a,Im(x));
            }
            else
            {
                // If we land here, then `op(x)` cannot be certified to be real at compile time, and `a != 0`. So we have no certificate for `a * op(x)` being real, and OpReturnRealQ should have returned `false`. So this is impossible.
                static_assert(Tools::DependentFalse<R>,"");
            }
        }

        template<Flag a_flag, Tools::Op op, typename R>
        TOOLS_FORCE_INLINE constexpr
        std::enable_if_t<
            VectorizableQ<R> && !OpReturnRealQ<a_flag,op,Cplx<R>,Cplx<R>>(),
            vec_T<2,R>
        >
        Op_vec( cref<Cplx<R>> a, cref<Cplx<R>> x )
        {
            using V_T = vec_T<2,R>;
            
            if constexpr ( (op == Tools::Op::Re) || (op == Tools::Op::ReTrans) )
            {
                // First case that op(x) is real.
                return Op_vec<a_flag,Tools::Op::Id>(a,Re(x));
            }
            else if constexpr ( (op == Tools::Op::Im) || (op == Tools::Op::ImTrans) )
            {
                // Second case that op(x) is real.
                return Op_vec<a_flag,Tools::Op::Id>(a,Im(x));
            }
            // From here one, we must assume that op(x) is complex.
            else if constexpr ( a_flag == Flag::Plus )
            {
                return Op_vec<Flag::Plus,op>(R(1),x);
            }
            else if constexpr ( a_flag == Flag::Minus )
            {
                return Op_vec<Flag::Minus,op>(R(-1),x);
            }
            else if constexpr ( a_flag == Flag::Generic )
            {
                if constexpr( op == Tools::Op::Id )
                {
                    V_T A_0 {  Re(a),  Im(a) };
                    V_T A_1 { -Im(a),  Re(a) };
                    
                    return A_0 * Re(x) + A_1 * Im(x);
                }
                else if constexpr( op == Tools::Op::Conj )
                {
                    V_T A_0 {  Re(a),  Im(a) };
                    V_T A_1 {  Im(a), -Re(a) };
                    
                    return A_0 * Re(x) + A_1 * Im(x);
                }
                else
                {
                    // We cannot arrive here because otherwise `op` where one of `Op::Re`, `Op::ReTrans`, `Op::Im`, or `Op::ImTrans`, and these cases have been seeved out above.
                    static_assert(Tools::DependentFalse<R>,"");
                }
            }
            else
            {
                // Only a_flag == Flag::Zero is possible here. But then `OpReturnRealQ` would have returned true, so which contradicts the assumptions in the template.
                static_assert(Tools::DependentFalse<R>,"");
            }
        }

    } // namespace Scalar
        
} // namespace Tools
