#pragma once

namespace Tensors
{
    
    enum class Compose : bool
    {
        Post = true,
        Pre  = false
    };
    
    enum class Inverse : bool
    {
        True  = true,
        False = false
    };
    
    template<IntQ Int_, Parallel_T parQ_>
    class Permutation final
    {
    public:
        
        using Int = Int_;
        
        static constexpr Parallel_T parQ = parQ_;
        
    protected:
        
        Int n;
        
        mutable Tensor1<Int,Int> p;
        mutable Tensor1<Int,Int> p_inv;
        
        mutable Tensor1<Int,Int> scratch;
        
        Int thread_count = 1;
        
        mutable bool is_trivial     = true;
        mutable bool p_computed     = true;
        mutable bool p_inv_computed = true;
        mutable bool is_valid       = false;
        
        static constexpr Int zero = Scalar::Zero<Int>;
        static constexpr Int one  = Scalar::One<Int>;
        
    public:
        
        Permutation( const Int n_, const Int thread_count_ )
        :   n              { n_                                           }
        ,   p              { iota<Int,Int>(n)                             }
        ,   p_inv          { iota<Int,Int>(n)                             }
        ,   scratch        { n                                            }
        ,   thread_count   { TruncateThreadCount<Int,parQ>(thread_count_) }
        ,   is_trivial     { true                                         }
        ,   p_computed     { true                                         }
        ,   p_inv_computed { true                                         }
        ,   is_valid       { n > Int(0)                                   }
        {}
        
        
        Permutation( Tensor1<Int,Int> && p_, const Inverse inverseQ, const Int thread_count_ )
        :   n              { p_.Size()                                    }
        ,   scratch        { n                                            }
        ,   thread_count   { TruncateThreadCount<Int,parQ>(thread_count_) }
        ,   is_trivial     { false                                        }
        ,   p_computed     { false                                        }
        ,   p_inv_computed { false                                        }
        {
            if( inverseQ == Inverse::True )
            {
                if( !PermutationQ(p_.data()) )
                {
                    eprint(ClassName()+"(): Input is not a permutation.");
                    is_valid = false;
                    return;
                }
                p_inv = std::move(p_);
                p_inv_computed = true;
                
                RequirePermutation();
                is_valid = true;
            }
            else
            {
                if( !PermutationQ(p_.data()) )
                {
                    eprint(ClassName()+"(): Input is not a permutation.");
                    is_valid = false;
                    return;
                }
                p = std::move(p_);
                p_computed = true;
                RequireInversePermutation();
                is_valid = true;
            }
        }
        
        template<IntQ J>
        Permutation( cptr<J> p_, const Int n_, const Inverse inverseQ, const Int thread_count_ )
        :   n              { n_                                           }
        ,   p              { n                                            }
        ,   p_inv          { n                                            }
        ,   scratch        { n                                            }
        ,   thread_count   { TruncateThreadCount<Int,parQ>(thread_count_) }
        {
            if( !PermutationQ(p_) )
            {
                eprint(ClassName()+"(): Input is not a permutation.");
                is_valid = false;
                return;
            }
            is_valid = true;
            
            if( inverseQ == Inverse::True )
            {
                SetInversePermutation(p_);
            }
            else
            {
                SetPermutation(p_);
            }
        }
        
        // Default constructor
        Permutation()
        :   n              { Int(0) }
        ,   p              { n      }
        ,   p_inv          { n      }
        ,   scratch        { n      }
        ,   thread_count   { Int(1) }
        ,   is_trivial     { true   }
        ,   p_computed     { true   }
        ,   p_inv_computed { true   }
        ,   is_valid       { true   }    // Yeah, we say that an empty permutation is valid! ^^
        {}
        
        // Destructor
        ~Permutation() = default;
        // Copy constructor
        Permutation( const Permutation & other ) = default;
        // Copy assignment operator
        Permutation & operator=( const Permutation & other ) = default;
        // Move constructor
        Permutation( Permutation && other ) = default;
        // Move assignment operator
        Permutation & operator=( Permutation && other ) = default;
        
    public:

        bool IsValid() const
        {
            return is_valid;
        }
        
        bool TrivialQ() const
        {
            return is_trivial;
        }

        void SetTrivial()
        {
            p.iota();
            p_inv.iota();
            is_trivial     = true;
            p_computed     = true;
            p_inv_computed = true;
        }
        
        Int Size() const
        {
            return n;
        }
        
        Int ThreadCount() const
        {
            return thread_count;
        }
        
        const Int & operator()( const Int i )
        {
            return p[i];
        }
        
        template<typename J>
        void SetPermutation( cptr<J> p_ )
        {
            // TODO: check p_ for triviality during copy.
            
            is_trivial = DoReduce<parQ>(
                [p_,this]( const Int i ) -> bool
                {
                    const Int p_i = int_cast<Int>(p_[i]);
                    
                    p[i] = p_i;
                    p_inv[p_i] = i;
                    
                    return (p_i == i);
                },
                []( bool v, mref<bool> r ) { r = r && v; }, true,
                n, thread_count
            );
            
            p_computed     = true;
            p_inv_computed = true;
        }

        
        // There is no nonconstant GetPermutation() because we want the error-free execution of the constructors to be a certificate.
        cref<Tensor1<Int,Int>> GetPermutation() const
        {
            return p;
        }
        
        template<typename J>
        void SetInversePermutation( cptr<J> p_inv_ )
        {
//            print("SetInversePermutation");
            // TODO: check p_inv_ for triviality during copy.
//            p_inv.Read(p_inv_);

            bool okayQ = DoReduce<parQ>(
                [p_inv_,this]( const Int i ) -> bool
                {
                    const Int p_inv_i = static_cast<Int>(p_inv_[i]);
                    
                    p_inv[i] = p_inv_i;
                    p[p_inv_i] = i;
                    
                    return (p_inv_i == i);
                },
                []( bool v, mref<bool> r ) { r = r && v; }, true,
                n, thread_count
            );
            
            if( !okayQ )
            {
                eprint(MethodName("SetInversePermutation") + ": Something went wrong here.");
            }
            
            p_computed     = true;
            p_inv_computed = true;
        }
        

        // There is no nonconstant GetPermutation() because we want the error-free execution of the constructors to be a certificate.
        cref<Tensor1<Int,Int>> GetInversePermutation() const
        {
            return p_inv;
        }

        
        void Invert( const Inverse inverseQ )
        {
            if( inverseQ == Inverse::True )
            {
                using std::swap; 
                swap( p, p_inv );
                swap( p_computed, p_inv_computed );
            }
        }
        
        void RequirePermutation()
        {
            if( !p_computed )
            {
                if ( p.Size() != n )
                {
                    p = Tensor1<Int,Int>(n);
                }
                
                Do<parQ>(
                    [=,this]( const Int i ) { p[p_inv[i]] = i; },
                    n, thread_count
                );
            }
        }
        
        void RequireInversePermutation()
        {
            if( !p_inv_computed  )
            {
                if ( p_inv.Size() != n )
                {
                    p_inv = Tensor1<Int,Int>(n);
                }
                Do<parQ>(
                    [this]( const Int i ) { p_inv[p[i]] = i; },
                    n, thread_count
                );
            }
        }
        
        
        void Compose( const Permutation & q, const Compose prepost )
        {
            TOOLS_PTIMER(timer,ClassName()+"::Compose");
            
            if( is_trivial )
            {
                if( q.is_trivial )
                {
                    // Do nothing.
                }
                else
                {
                    p.Read( q.GetPermutation().data() );
                    p_inv.Read( q.GetInversePermutation().data() );

                    is_trivial     = q.TrivialQ();
                    p_computed     = true;
                    p_inv_computed = true;
                }
            }
            else
            {
                if( q.is_trivial )
                {
                    // Do nothing.
                }
                else
                {
                    //Careful! This invalidates pointers!
                    
                    p_computed     = true;
                    p_inv_computed = true;
                    
//                    cptr<Int> q_    = q.GetPermutation().data();
//                    cptr<Int> q_inv = q.GetInversePermutation().data();
                    
                    const Int * restrict a     = nullptr;
                    const Int * restrict b     = nullptr;
                    const Int * restrict a_inv = nullptr;
                    const Int * restrict b_inv = nullptr;
                    
                    if( prepost == Compose::Post )
                    {
                        // i.e. do p[i] <- p[q[i]];
                        
                        a =   GetPermutation().data();
                        b = q.GetPermutation().data();
                        
                        a_inv =   GetInversePermutation().data();
                        b_inv = q.GetInversePermutation().data();
                    }
                    else
                    {
                        // i.e. do p[i] <- q[p[i]];
                        
                        a = q.GetPermutation().data();
                        b =   GetPermutation().data();
                        
                        a_inv = q.GetInversePermutation().data();
                        b_inv =   GetInversePermutation().data();
                    }
                    
                    is_trivial = DoReduce<parQ>(
                        [a,b,this]( const Int i ) -> bool
                        {
                            scratch[i] = a[b[i]];
                            
                            return (scratch[i] == i);
                        },
                        []( bool v, mref<bool> r ) { r = r && v; }, true,
                        n, thread_count
                    );
                        
                    swap(p,scratch);
                    
                    //post
                    Do<parQ>(
                        [=,this]( const Int i ) { scratch[i] = b_inv[a_inv[i]]; },
                        n, thread_count
                    );
                    
                    swap(p_inv,scratch);
                }
            }
        }

        template<
            Size_T COLS = VarSize,
            Op opx = Op::Id, Op opy = Op::Id,
            typename a_T, typename X_T, typename b_T, typename Y_T
        >
        void PermuteCombine(
            const a_T alpha, cptr<X_T> X, const Int ldX,
            const b_T beta,  mptr<Y_T> Y, const Int ldY,
            const Inverse inverseQ,
            Int cols = COLS
        )
        {
            // Computes
            //
            //   Y[i] = alpha * X[ p[i] ] + beta * Y[i]
            //
            // or
            //
            //   Y[i] = alpha * X[ q[i] ] + beta * Y[i]
            
            using F_T = Scalar::Flag;
            
            std::string tag = ClassName()+"::PermuteCombine"
                + "," + ToString(COLS)
                + "," + ToString(parQ)
                + "," + ToString(opx)
                + "," + ToString(opy)
                + "," + TypeName<a_T>
                + "," + TypeName<X_T>
                + "," + TypeName<b_T>
                + "," + TypeName<Y_T>
                + ">(" + (inverseQ == Inverse::True ? "inv," : "id," )
                + ToString(cols) + ")";
            
            TOOLS_PTIMER(timer,tag);
            
            if( cols == Int(0) )
            {
                wprint( tag + ": cols == 0. Doing nothing." );
            }

            // TODO: Special case ldX = 1 and ldY = 1
            
            // TODO: Special case trivial permutation -> combine_matrices
            
            Invert( inverseQ );
            
            cptr<Int> r = GetPermutation().data();

            auto do_job = [=,this]<F_T a_flag,F_T b_flag >()
            {
                if( !this->is_trivial )
                {
                    Do<parQ>(
                        [=]( const Int i )
                        {
                            combine_buffers<a_flag,b_flag,COLS,Sequential,opx,opy>
                            ( 
                                alpha, &X[ldX * r[i]], beta, &Y[ldY * i], cols
                            );
                        },
                        n, thread_count
                    );
                }
                else
                {
                    combine_matrices<a_flag,b_flag,VarSize,COLS,parQ,opx,opy>
                    (
                        alpha, X, ldX, beta, Y, ldY, n, cols, thread_count
                    );
                }
            };
            
            if( alpha == a_T(1) )
            {
                constexpr F_T a_flag = F_T::Plus;
                
                if( beta == b_T(1) )
                {
                    constexpr F_T b_flag = F_T::Plus;
                    do_job.template operator()<a_flag,b_flag>();
                }
                else if ( beta == b_T(0) )
                {
                    constexpr F_T b_flag = F_T::Zero;
                    do_job.template operator()<a_flag,b_flag>();
                }
//                else if ( beta == b_T(-1) )
//                {
//                    constexpr F_T beta_flag = F_T::Minus;
//                    do_job.template operator()<a_flag,b_flag>();
//                }
                else
                {
                    constexpr F_T b_flag = F_T::Generic;
                    do_job.template operator()<a_flag,b_flag>();
                }
            }
            else if ( alpha == a_T(0) )
            {
                constexpr F_T a_flag = F_T::Zero;
                
                if( beta == b_T(1) )
                {
                    constexpr F_T b_flag = F_T::Plus;
                    do_job.template operator()<a_flag,b_flag>();
                }
                else if ( beta == b_T(0) )
                {
                    constexpr F_T b_flag = F_T::Zero;
                    do_job.template operator()<a_flag,b_flag>();
                }
//                else if ( beta == b_T(-1) )
//                {
//                    constexpr F_T beta_flag = F_T::Minus;
//                    do_job.template operator()<a_flag,b_flag>();
//                }
                else
                {
                    constexpr F_T b_flag = F_T::Generic;
                    do_job.template operator()<a_flag,b_flag>();
                }
            }
//            else if ( alpha == a_T(-1) )
//            {
//                constexpr F_T a_flag = F_T::Minus;
//                
//                if( beta == b_T(1) )
//                {
//                    constexpr F_T b_flag = F_T::Plus;
//                    do_job.template operator()<a_flag,b_flag>();
//                }
//                else if ( beta == b_T(0) )
//                {
//                    constexpr F_T b_flag = F_T::Zero;
//                    do_job.template operator()<a_flag,b_flag>();
//                }
////                else if ( beta == b_T(-1) )
////                {
////                    constexpr F_T beta_flag = F_T::Minus;
////                    do_job.template operator()<a_flag,b_flag>();
////                }
//                else
//                {
//                    constexpr F_T b_flag = F_T::Generic;
//                    do_job.template operator()<a_flag,b_flag>();
//                }
//            }
            else
            {
                constexpr F_T a_flag = F_T::Generic;
                
                if( beta == b_T(1) )
                {
                    constexpr F_T b_flag = F_T::Plus;
                    do_job.template operator()<a_flag,b_flag>();
                }
                else if ( beta == b_T(0) )
                {
                    constexpr F_T b_flag = F_T::Zero;
                    do_job.template operator()<a_flag,b_flag>();
                }
//                else if ( beta == b_T(-1) )
//                {
//                    constexpr F_T beta_flag = F_T::Minus;
//                    do_job.template operator()<a_flag,b_flag>();
//                }
                else
                {
                    constexpr F_T b_flag = F_T::Generic;
                    do_job.template operator()<a_flag,b_flag>();
                }
            }
            
            Invert( inverseQ );
        }
        
        template<Size_T COLS = VarSize, typename X_T, typename Y_T>
        void Permute(
            cptr<X_T> X, const Int ldX,
            mptr<Y_T> Y, const Int ldY,
            const Inverse inverseQ,
            Int cols = ( (COLS>VarSize) ? int_cast<Int>(COLS) : Scalar::One<Int> )
        )
        {
            // Permute X chunkwise into Y, i.e., Y[ldY*i+k] <- X[ldX*p[i]+k];
            
            TOOLS_PTIMER(timer,ClassName()+"::Permute<" + TypeName<X_T> + "," + TypeName<Y_T> + ">(" + (inverseQ == Inverse::True ? "inv," : "id," ) + ToString(cols) + ")");
            
            if( !is_trivial )
            {
                Invert( inverseQ );
                
                cptr<Int> r = GetPermutation().data();

                if( cols == Scalar::One<Int> )
                {
                    // TODO: Is there any merit in this specialization?
                    // TODO: Shouldn't copy_buffer handle that appropriately?
                    
                    if( (ldX == Scalar::One<Int>) && (ldY == Scalar::One<Int>) )
                    {
                        Do<parQ>(
                            [=,this]( const Int i )
                            {
                                Y[i] = static_cast<Y_T>(X[r[i]]);
                            },
                            n, thread_count
                        );
                    }
                    else
                    {
                        Do<parQ>(
                            [=,this]( const Int i )
                            {
                                Y[ldY * i] = static_cast<Y_T>(X[ldX * r[i]]);
                            },
                            n, thread_count
                        );
                    }
                }
                else
                {
                    Do<parQ>(
                        [=,this]( const Int i )
                        {
                            copy_buffer<COLS>( &X[ldX * r[i]], &Y[ldY * i], cols );
                        },
                        n, thread_count
                    );
                }
                
                Invert( inverseQ );
            }
            else
            {
                if( (ldX == cols) && (ldY == cols) )
                {
                    copy_buffer<VarSize,parQ>( X, Y, n*cols, thread_count );
                }
                else
                {
                    copy_matrix<VarSize,COLS,parQ>( X, ldX, Y, ldY, n, cols, thread_count );
                }
            }
        }
        
        template<Size_T COLS = VarSize, typename X_T, typename Y_T>
        void Permute(
            cptr<X_T> X,
            mptr<Y_T> Y,
            const Inverse inverseQ,
            Int cols = Scalar::One<Int>
        )
        {
            Permute<COLS>( X, cols, Y, cols, inverseQ, cols );
        }
        
        template<typename X_T, typename Y_T>
        void Permute( const Tensor1<X_T,Int> & X, Tensor1<Y_T,Int> & Y, const Inverse inverseQ )
        {
            if( X.Size() != n )
            {
                eprint(ClassName()+"::Permute: First input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            if( Y.Size() != n )
            {
                eprint(ClassName()+"::Permute: Second input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            Permute<1>( X.data(), Y.data(), inverseQ, Int(1) );
        }
        
        template<typename X_T, typename Y_T>
        void Permute( const Tensor2<X_T,Int> & X, Tensor2<Y_T,Int> & Y, const Inverse inverseQ )
        {
            if( X.Size() != n )
            {
                eprint(ClassName()+"::Permute: First input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            if( Y.Size() != n )
            {
                eprint(ClassName()+"::Permute: Second input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            if( X.Dim(1) != Y.Dim(1) )
            {
                eprint(ClassName()+"::Permute: Number of columns of input arrays do not coincide. Doing nothing.");
                return;
            }
            
            Permute<VarSize>( X.data(), Y.data(), inverseQ, X.Dim(1) );
        }
        
        
        // Somewhat dangerous. Use this only if you know what you are doing!
        Tensor1<Int,Int> & Scratch()
        {
            return scratch;
        }
        
        // Somewhat dangerous. Use this only if you know what you are doing!
        void SwapScratch( Inverse inverseQ )
        {
            is_trivial = true;
            
            if( inverseQ == Inverse::False )
            {
                swap( p, scratch );
                
                is_trivial = DoReduce<parQ>(
                    [this]( const Int i ) -> bool
                    {
                        const Int p_i = p[i];
                        
                        p_inv[p_i] = i;
                        
                        return (p_i == i);
                    },
                    []( bool v, mref<bool> r ) { r = r && v; }, true,
                    n, thread_count
                );
            }
            else
            {
                swap( p_inv, scratch );
                
                is_trivial = DoReduce<parQ>(
                    [=,this]( const Int i ) -> bool
                    {
                        const Int p_inv_i = p_inv[i];
                        
                        p[p_inv_i] = i;
                        
                        return (p_inv_i == i);
                    },
                    []( bool v, mref<bool> r ) { r = r && v; }, true,
                    n, thread_count
                );
            }

            p_computed     = true;
            p_inv_computed = true;
        }
        
        
        template<typename J>
        bool PermutationQ( cptr<J> p_ ) const
        {
            TOOLS_PTIMER(timer, MethodName("PermutationQ"));
            if( (n == zero) || (n == one ) )
            {
                return true;
            }
            
            scratch.SetZero();
            
            for( Int i = 0; i < n; ++i )
            {
                const Int p_i = int_cast<Int>(p_[i]);
                
                if( (p_i < zero) || (p_i >= n) )
                {
                    wprint(MethodName("PermutationQ") + ": Input list p has value p["+ToString(i)+"] = "+ToString(p_i)+" out of range [0,"+ToString(n)+"[!");
                    return false;
                }
                else
                {
                    ++scratch[p_i];
                }
            }

            const auto [m_0, m_1] = minmax_buffer( scratch.data(), n );

            if( m_0 != one )
            {
                eprint(MethodName("PermutationQ") + ": Input does not attain all values in range!");
            }
            
            if( m_1 != one )
            {
                eprint(MethodName("PermutationQ") + ": Input has duplicates!");
            }
            
            return ( m_0 == one ) && ( m_1 );
        }
        
        bool PermutationQ() const
        {
            if( PermutationQ(p.data()) )
            {
                if( PermutationQ(p_inv.data()) )
                {
                    return true;
                }
                else
                {
                    eprint(MethodName("PermutationQ") + ": field p_inv is not a permutation!");
                    return false;
                }
            }
            else
            {
                eprint(MethodName("PermutationQ") +": field p is not a permutation!");
                return false;
            }
        }
        
        
        template<IntQ ExtInt, typename PRNGT_T>
        static Permutation RandomPermutation(
            const ExtInt n,
            const ExtInt thread_count,
            mref<PRNGT_T> random_engine
        )
        {
            if( n <= ExtInt(0) ) { return Permutation(); }
            
            Tensor1<Int,Int> a(n);
            a.iota();
            
            std::shuffle(&a[Int(0)],&a[Int(n)],random_engine);
            
            return Permutation( std::move(a), Inverse::False, int_cast<Int>(thread_count) );
        }
                       
    public:

        static std::string MethodName( const std::string & tag )
        {
            return ClassName() + "::" + tag;
        }
        
        static std::string ClassName()
        {
            return std::string("Permutation")+"<"+TypeName<Int>+">";
        }

    }; // class Permutation
    
    
    
    template<bool P_TrivialQ, bool Q_TrivialQ, bool SortQ, IntQ Int, IntQ LInt,
        Parallel_T P_parQ, Parallel_T Q_parQ
    >
    Tensor1<LInt,LInt> permutePatternCSR(
        mref<Tensor1<LInt,Int>> outer,
        mref<Tensor1<Int,LInt>> inner,
        cref<Permutation<Int,P_parQ>>  P,  // row    permutation
        cref<Permutation<Int,Q_parQ>>  Q,  // column permutation
        const LInt nnz
    )
    {
        TOOLS_PTIMER(timer,std::string("PermutePatternCSR<") + TypeName<Int> + "," + TypeName<LInt> + ">");
        
        const Int m = P.Size();

        const Int thread_count = P.ThreadCount();
        
        cptr<Int> p     = P.GetPermutation().data();
        cptr<Int> q_inv = Q.GetInversePermutation().data();

        Tensor1<LInt, Int> new_outer ( m+1 );
        Tensor1< Int,LInt> new_inner ( nnz );
        
        Tensor1<LInt,LInt> perm ( nnz );

        if constexpr ( P_TrivialQ && Q_TrivialQ )
        {
            perm.iota();
            return perm;
        }
        
        if constexpr ( P_TrivialQ )
        {
            swap( outer, new_outer );
        }
        else
        {
            new_outer[0] = 0;
            
            Do<P_parQ>(
                [&]( const Int i )
                {
                    const Int p_i = p[i];
                    new_outer[i+1] = static_cast<LInt>(outer[p_i+Int(1)] - outer[p_i]);
                },
                m, thread_count
            );

            Accumulate<P_parQ>( new_outer.data(), m + Int(1), thread_count );
        }
        
        mptr<LInt> scratch = perm.data();

        JobPointers<Int> job_ptr ( m, new_outer.data(), thread_count );
        
        Do<P_parQ>(
            [&,scratch]( const Int thread )
            {
                TwoArraySort<Int,LInt,Int> S;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt begin = outer[ P_TrivialQ ? i : p[i] ];
                    
                    const LInt new_begin = new_outer[i  ];
                    const LInt new_end   = new_outer[i+1];
                    
                    const LInt k_max = new_end - new_begin;
                    
                    if constexpr ( Q_TrivialQ )
                    {
                        copy_buffer( &inner[begin], &new_inner[new_begin], k_max );
                        
                        for( LInt k = 0; k < k_max; ++k )
                        {
                            scratch[new_begin+k] = begin+k;
                        }
                    }
                    else
                    {
                        for( LInt k = 0; k < k_max; ++k )
                        {
                            new_inner[new_begin+k] = q_inv[inner[begin+k]];
                            scratch  [new_begin+k] = begin+k;
                        }
                        
                        if constexpr ( SortQ )
                        {
                            S( &new_inner[new_begin], &scratch [new_begin], static_cast<Int>(k_max) );
                        }
                    }
                }
            },
            thread_count, thread_count
        );
        
        swap( outer, new_outer );
        swap( inner, new_inner );

        return perm;
    }
    
    template<IntQ Int, IntQ LInt, Parallel_T P_parQ, Parallel_T Q_parQ>
    Tensor1<LInt,LInt> PermutePatternCSR(
        mref<Tensor1<LInt,Int>> outer,
        mref<Tensor1<Int,LInt>> inner,
        cref<Permutation<Int,P_parQ>>  P,  // row    permutation
        cref<Permutation<Int,Q_parQ>>  Q,  // column permutation
        const LInt nnz,
        bool sort = true // Whether to restore row-wise ordering (as demanded by CSR).
    )
    {
        // returns
        // i  ) The permuted array "outer".
        // ii ) The permuted array "inner".
        // iii) The permutation that has to be applied to the nonzero values.
        
        if( sort )
        {
            return permutePatternCSR<false,false,true>(outer,inner,P,Q,nnz);
        }
        else
        {
            return permutePatternCSR<false,false,false>(outer,inner,P,Q,nnz);
        }
    }
    
    
} // namespace Tensors
