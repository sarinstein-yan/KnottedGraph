#pragma once

namespace Tensors
{
    namespace Sparse
    {
        // TODO: This is a very incomplete class that can only hold sorted indices and values. More work has to be done on this.
        template<typename Scal_, IntQ Int_>
        class Vector final
        {
            
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            
        protected:
            
            Int n = 0;
            Int thread_count = 1;
            Tensor1<Int ,Int> idx;  // Needs to be sorted!
            Tensor1<Scal,Int> val;
            Scal default_value = 0;
            
        public:
            
            template<IntQ I_0, IntQ I_1, IntQ I_2, typename S>
            Vector(
               const I_0 n_,
               const I_1 nnz_,
               const I_2 thread_count_  = 1,
               const S   default_value_ = 0
            )
            :   n             { int_cast<Int>(n_)                 }
            ,   idx           { int_cast<Int>(nnz_)               }
            ,   val           { int_cast<Int>(nnz_)               }
            ,   thread_count  { int_cast<Int>(thread_count_)      }
            ,   default_value { static_cast<Scal>(default_value_) }
            {}
            
            template<typename S, IntQ J_0, IntQ I_0, IntQ I_1, IntQ I_2>
            Vector(
                const J_0 * const idx_,  // Needs t be sorted!
                const S   * const val_,
                const I_0 n_,
                const I_1 nnz_,
                const I_2 thread_count_  = 1,
                const S   default_value_ = 0
            )
            :   n    { int_cast<Int>(n_)         }
            ,   idx  { idx_, int_cast<Int>(nnz_) }
            ,   val  { val_, int_cast<Int>(nnz_) }
            ,   thread_count { int_cast<Int>(thread_count_) }
            ,   default_value { static_cast<Scal>(default_value_) }
            {
                static_assert(ArithmeticQ<S>,"");
            }
            
            // Default constructor
            Vector() = default;
            // Destructor
            ~Vector() = default;
            // Copy constructor
            Vector( const Vector & other ) = default;
            // Copy assignment operator
            Vector & operator=( const Vector & other ) = default;
            // Move constructor
            Vector( Vector && other ) = default;
            // Move assignment operator
            Vector & operator=( Vector && other ) = default;
            
            
            friend void swap (Vector & A, Vector & B ) noexcept
            {
                using std::swap;
                
                swap( A.n,            B.n            );
                swap( A.thread_count, B.thread_count );
                swap( A.idx,          B.idx          );
                swap( A.val,          B.val          );
            }
            
        public:

            
            Int Size() const
            {
                return n;
            }
            
            Int ExplicitCount() const
            {
                return idx.Size();
            }
            
            Scal DefaultValue() const
            {
                return default_value;
            }
            
            mref<Tensor1<Scal,Int>> Indices()
            {
                return idx;
            }
            
            cref<Tensor1<Scal,Int>> Indices() const
            {
                return idx;
            }
            
            mref<Tensor1<Scal,Int>> Values()
            {
                return val;
            }
            
            cref<Tensor1<Scal,Int>> Values() const
            {
                return val;
            }
            
            mref<Tensor1<Scal,Int>> Value()
            {
                return val;
            }
            
            cref<Tensor1<Scal,Int>> Value() const
            {
                return val;
            }
            
            mref<Scal> Value( const Int k )
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < Int(0) || k >= val.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return val[k];
            }
            
            cref<Scal> Value( const Int k ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < Int(0) || k >= val.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return val[k];
            }
            
        public:
            
            static std::string MethodName( const std::string & tag )
            {
                return ClassName() + "::" + tag;
            }
            
            static std::string ClassName()
            {
                return std::string("Sparse::Vector")
                    + "<" + TypeName<Scal>
                    + "," + TypeName<Int>
                    + ">";
            }
            
        }; // Vector
        
        
        
        /*!@brief This effectively reads odd the portions of the sparse vectors `u` and `v` given by `{u_i_begin,...,u_i_end-1}` and `{ v_i_begin,...,v_i_end-1}` and computes their dot product without doing any actual copy.
         * Binary search (`FindRange`) is used to find the range in `u_idx` and `v_idx` that correspond to the indices `{u_i_begin,...,u_i_end-1}` and `{v_i_begin,...,v_i_end-1}`, so `u_idx` and `v_idx` are required to be sorted.
         */
        
        template<
            typename u_Scal, typename u_Int,
            typename v_Scal, typename v_Int,
            typename Scal = decltype( u_Scal(0) *  v_Scal(0))
        >
        static Scal VectorPartialDot(
            cptr<u_Int>  u_idx,
            cptr<u_Scal> u_val,
            const u_Scal u_default,
            const u_Int  u_n,
            const u_Int  u_nnz,
            const u_Int  u_i_begin,
            const u_Int  u_i_end,
            cptr<v_Int>  v_idx,
            cptr<v_Scal> v_val,
            const u_Scal v_default,
            const v_Int  v_n,
            const v_Int  v_nnz,
            const v_Int  v_i_begin,
            const v_Int  v_i_end
        )
        {
            (void)u_n;
            (void)v_n;
            
            if( u_i_begin >= u_i_end ) { return 0; }
            if( v_i_begin >= v_i_end ) { return 0; }
            
            if( std::cmp_not_equal(u_i_end - u_i_begin,v_i_end - v_i_begin) )
            {
                eprint(std::string("Sparse::VectorPartialDot")+": u_i_end - u_i_begin != v_i_end - v_i_begin.");
                return 0;
            }
            
//            TOOLS_DUMP(u_idx);
//            TOOLS_DUMP(u_val);
//            TOOLS_DUMP(u_nnz);
//            TOOLS_DUMP(v_idx);
//            TOOLS_DUMP(v_val);
//            TOOLS_DUMP(v_nnz);
            
//            TOOLS_DUMP(u_i_begin);
//            TOOLS_DUMP(u_i_end  );
//            TOOLS_DUMP(v_i_begin);
//            TOOLS_DUMP(v_i_end  );
//            
//            TOOLS_DUMP(u_default);
//            TOOLS_DUMP(v_default);
//            
//            TOOLS_DUMP(u_nnz);
//            TOOLS_DUMP(v_nnz);
//            
//            TOOLS_DUMP(u_idx);
//            TOOLS_DUMP(u_val);
//            
//            TOOLS_DUMP(v_idx);
//            TOOLS_DUMP(v_val);
            
            // TODO: Check cost here if u_i_begin == 0 and u_i_end == u_n.
            auto [u_p_begin,u_p_end] = ( u_idx == nullptr ) || ( u_val == nullptr )
                                     ? std::pair{u_nnz,u_nnz}
                                     : FindRange(
                                           u_idx,u_nnz,u_i_begin,u_i_end
                                       );
            
//            TOOLS_DUMP(u_p_begin);
//            TOOLS_DUMP(u_p_end);
            
            // TODO: Check cost here if v_i_begin == 0 and v_i_end == v_n.
            auto [v_p_begin,v_p_end] = ( v_idx == nullptr ) || ( v_val == nullptr )
                                     ? std::pair{v_nnz,v_nnz}
                                     : FindRange(
                                           v_idx,v_nnz,v_i_begin,v_i_end
                                       );
            
//            TOOLS_DUMP(v_p_begin);
//            TOOLS_DUMP(v_p_end);
            
            // Caution: One might be tempted to use Details::Dot from here on, but we have to apply a shift of the indices in order to test for equality. Also Details::Dot does not account for nonzero default values.

            const Scal u_def = static_cast<Scal>(u_default);
            const Scal v_def = static_cast<Scal>(v_default);
            
            const bool u_defQ = (u_def != Scal(0));
            const bool v_defQ = (v_def != Scal(0));
            
            Scal   sum = 0;
            Size_T uv_counter = 0; // Counts the number of explicit position matches.

            // Pointers into u_idx and v_idx.
            u_Int u_p = u_p_begin;
            v_Int v_p = v_p_begin;
            
            // Caution: We must not terminate here if u_p >= u_p_end or v_p >= v_p_end as this won't treat nonzero default values correctly.
            
            if( (u_p < u_p_end) && (v_p < v_p_end) )
            {
                u_Int u_i = u_idx[u_p] - u_i_begin; // Corrects non-0-basedness!
                v_Int v_i = v_idx[v_p] - v_i_begin; // Corrects non-0-basedness!

                while( true )
                {
                    if ( u_i < v_i )
                    {
                        // We put an if-check here to give branch prediction a chance for optimization. (Also, v_defQ is marked const, so maybe the compiler creates appropriate branches at compile time.)
                        if ( v_defQ )
                        {
                            sum += static_cast<Scal>(u_val[u_p]) * v_def;
                        }
                        ++u_p;
                        if ( u_p >= u_p_end ) { break; }
                        u_i = u_idx[u_p] - u_i_begin; // Corrects non-0-basedness!
                    }
                    else if ( u_i > v_i )
                    {
                        // We put an if-check here to give branch prediction a chance for optimization. (Also, u_defQ is marked const, so maybe the compiler creates appropriate branches at compile time.)
                        if ( u_defQ )
                        {
                            sum += u_def * static_cast<Scal>(v_val[v_p]);
                        }
                        ++v_p;
                        if ( v_p >= v_p_end ) { break; }
                        v_i = v_idx[v_p] - v_i_begin; // Corrects non-0-basedness!
                    }
                    else // if ( u_i == v_i )
                    {
                        ++uv_counter;
                        sum += static_cast<Scal>(u_val[u_p]) * static_cast<Scal>(v_val[v_p]);
                        
                        ++u_p;
                        ++v_p;
                        if ( (u_p >= u_p_end) || (v_p >= v_p_end) ) { break; }
                        u_i = u_idx[u_p] - u_i_begin; // Corrects non-0-basedness!
                        v_i = v_idx[v_p] - v_i_begin; // Corrects non-0-basedness!
                    }
                }
            }
            
            // If v_def != 0, then we have to handle the unvisited explicit values of u.
            if ( v_defQ )
            {
                while ( u_p < u_p_end )
                {
                    sum += static_cast<Scal>(u_val[u_p]) * v_def;
                    ++u_p;
                }
            }
            
            // If u_def != 0, then we have to handle the unvisited explicit values of v.
            if ( u_defQ )
            {
                while ( v_p < v_p_end )
                {
                    sum += u_def * static_cast<Scal>(v_val[v_p]);
                    ++v_p;
                }
            }
            
            if ( u_defQ && v_defQ )
            {
                const Size_T u_i_count = ToSize_T(u_i_end - u_i_begin);
//                const Size_T v_i_count = ToSize_T(v_i_end - v_i_begin);
                
                const Size_T u_p_count = ToSize_T(u_p_end - u_p_begin);
                const Size_T v_p_count = ToSize_T(v_p_end - v_p_begin);
                
//                // The number of cases where an explicit position of u was not matched by an explicit position of v.
//                const Size_T u_p_unmatched_count = u_p_count - uv_counter;
                
                // The number of cases where an explicit position of v was not matched by an explicit position of u.
                const Size_T v_p_unmatched_count = v_p_count - uv_counter;
                
//                TOOLS_DUMP(sum);
                
                const Size_T u_def_pair_count = u_i_count - u_p_count - v_p_unmatched_count;
                
                sum += u_def * v_def * static_cast<Scal>(u_def_pair_count);
            }
            
            return sum;
        }
        
        
        template<
            typename u_Scal, IntQ Int, typename Scal = u_Scal
        >
        static Scal VectorPartialSum(
            cptr<Int>    u_idx,
            cptr<u_Scal> u_val,
            const u_Scal u_default,
            const Int    u_n,
            const Int    u_nnz,
            const Int    u_i_begin,
            const Int    u_i_end
        )
        {
            Int  * v_idx = nullptr;
            Scal * v_val = nullptr;
            
            return VectorPartialDot<u_Scal,Int,Scal,Int,Scal>(
                u_idx, u_val, u_default, u_n, u_nnz , u_i_begin, u_i_end,
                v_idx, v_val, Scal(1)  , u_n, Int(0), u_i_begin, u_i_end
            );
        }
        
        template<
            typename u_Scal, IntQ Int, typename Scal = u_Scal
        >
        static Scal VectorTotal(
            cptr<Int>    u_idx,
            cptr<u_Scal> u_val,
            const u_Scal u_default,
            const Int    u_n,
            const Int    u_nnz
        )
        {
            Int  * v_idx = nullptr;
            Scal * v_val = nullptr;
            
            return VectorPartialDot<u_Scal,Int,Scal,Int,Scal>(
                u_idx, u_val, u_default, u_n, u_nnz , Int(0), Int(u_n),
                v_idx, v_val, Scal(1)  , u_n, Int(0), Int(0), Int(u_n)
            );
        }
        
    } // namespace Sparse
    
    
    /*!@brief This effectively reads odd the portions of the sparse vectors `u` and `v` given by `{u_i_begin,...,u_i_end-1}` and `{ v_i_begin,...,v_i_end-1}` and computes their dot product without doing any actual copy.
     */
    
    template<
        typename u_Scal, typename u_Int,
        typename v_Scal, typename v_Int,
        typename Scal = decltype( u_Scal(0) * v_Scal(0))
    >
    Scal PartialDot(
        cref<Sparse::Vector<u_Scal,u_Int>> u,
        const u_Int  u_i_begin,
        const u_Int  u_i_end,
        cref<Sparse::Vector<v_Scal,v_Int>> v,
        const v_Int  v_i_begin,
        const v_Int  v_i_end
    )
    {
        return Sparse::VectorPartialDot(
            u.Indices().data(),
            u.Values().data(),
            u.DefaultValue(),
            u.Size(), u.ExplicitCount(), u_i_begin, u_i_end,
            v.Indices().data(),
            v.Values().data(),
            v.DefaultValue(),
            v.Size(), v.ExplicitCount(), v_i_begin, v_i_end
        );
    }
    
    template<
        typename u_Scal, IntQ Int,
        typename Scal = u_Scal
    >
    Scal PartialSum(
        cref<Sparse::Vector<u_Scal,Int>> u,
        const Int  u_i_begin,
        const Int  u_i_end
    )
    {
        return Sparse::VectorPartialSum<u_Scal,Int,Scal>(
            u.Indices().data(),
            u.Values().data(),
            u.DefaultValue(),
            u.Size(), u.ExplicitCount(), u_i_begin, u_i_end
        );
    }
    
    template<
        typename u_Scal, IntQ u_Int,
        typename v_Scal, IntQ v_Int,
        typename Scal = decltype( u_Scal(0) * v_Scal(0) )
    >
    Scal Dot(
        cref<Sparse::Vector<u_Scal,u_Int>> u,
        cref<Sparse::Vector<v_Scal,v_Int>> v
    )
    {
        return Sparse::template VectorPartialDot<u_Scal,u_Int,v_Scal,v_Int,Scal>(
            u.Indices().data(),
            u.Values().data(),
            u.DefaultValue(),
            u.Size(), u.ExplicitCount(), u_Int(0), u.Size(),
            v.Indices().data(),
            v.Values().data(),
            v.DefaultValue(),
            v.Size(), v.ExplicitCount(), v_Int(0), v.Size()
        );
    }
    
    template<
        typename u_Scal, IntQ Int,
        typename Scal = u_Scal
    >
    Scal Total(
        cref<Sparse::Vector<u_Scal,Int>> u
    )
    {
        return Sparse::VectorTotal<u_Scal,Int,Scal>(
            u.Indices().data(),
            u.Values().data(),
            u.DefaultValue(),
            u.Size(), u.ExplicitCount()
        );
    }
    
} // namespace Tensors
