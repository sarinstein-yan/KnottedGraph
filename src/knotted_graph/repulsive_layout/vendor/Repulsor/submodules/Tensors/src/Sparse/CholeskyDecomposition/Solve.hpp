public:

// TODO: Collision of these temporary parQ with the class's parQ.

// TODO: Overloads for UpperSolve

// TODO: Overloads for LowerSolve

//###########################################################
//####          Solve
//###########################################################


    // The most general Solve routine that is exposed to the user.
    // It loads the inputs, runs the actual solver, and writes the outputs.

    // Evaluate X = alpha * op(A)^{-1} . B + beta * X
    template<
        Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id,
        typename a_T, typename B_T, typename b_T, typename X_T
    >
    void Solve(
        const a_T alpha, cptr<B_T> B,     const Int ldB,
        const b_T beta,  mptr<X_T> X_out, const Int ldX,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        const std::string tag = ClassName()+"::Solve"
            + "<" + ToString(VarSize)
            + "," + ToString(parQ)
            + "," + ToString(op)
            + "," + TypeName<a_T>
            + "," + TypeName<B_T>
            + "," + TypeName<b_T>
            + "," + TypeName<X_T>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        TOOLS_PTIMER(timer,tag);
        // No problem if X_out and B overlap, since we load B into X anyways.
        
        if constexpr ( NRHS > VarSize )
        {
            if( nrhs_ != NRHS )
            {
                eprint( tag + ": (NRHS > VarSize) && (nrhs_ != NRHS). Aborting.");
                return;
            }
        }
        
        ReadRightHandSide<NRHS>( B, ldB, nrhs_ );
        
        if( nrhs == ione )
        {
            if( thread_count > Int(1) )
            {
                SN_Solve<false,parQ,op>();
            }
            else
            {
                SN_Solve<false,Sequential,op>();
            }
        }
        else
        {
            if( thread_count > Int(1) )
            {
                SN_Solve<true,parQ,op>();
            }
            else
            {
                SN_Solve<true,Sequential,op>();
            }
        }

        WriteSolution<NRHS>( alpha, beta, X_out, ldX );
    }

    // Evaluate X = op(A)^{-1} . B
    template<
        Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id,
        typename B_T, typename X_T
    >
    void Solve(
        cptr<B_T> B,
        mptr<X_T> X_out,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        Solve<NRHS,parQ,op>(
            Scalar::One <X_T>, B,     nrhs_,
            Scalar::Zero<X_T>, X_out, nrhs_,
            nrhs_
        );
    }

    // Evaluate X = alpha * op(A)^{-1} . B + beta * X
    template<
        Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id,
        typename a_T, typename B_T, typename b_T, typename X_T,
        typename B_I, typename X_I
    >
    void Solve(
        const a_T alpha, cref<Tensor2<B_T,B_I>> B,
        const b_T beta,  mref<Tensor2<X_T,X_I>> X_out
    )
    {
        
        if ( B.Dim(0) < RowCount() )
        {
            eprint( ClassName()+"::Solve: B.Dim(0) < RowCount(). Aborting");
            return;
        }
        
        if ( X_out.Dim(0) < RowCount() )
        {
            eprint( ClassName()+"::Solve: X.Dim(0) < RowCount(). Aborting");
            return;
        }
        
        if ( B.Dim(1) != X_out.Dim(1) )
        {
            eprint( ClassName()+"::Solve: B.Dim(1) != B.Dim(1).");
            return;
        }
        
        const Int ldB = int_cast<Int>(B.Dim(1));
        const Int ldX = int_cast<Int>(X_out.Dim(1));
        
        Solve<NRHS,parQ,op>(
            alpha, B.data(),     ldB,
            beta,  X_out.data(), ldX,
            ldX
        );
    }

    // Evaluate X = alpha * op(A)^{-1} . B + beta * X
    template<
        Parallel_T parQ = Sequential, Op op = Op::Id,
        typename a_T, typename B_T, typename b_T, typename X_T,
        typename B_I, typename X_I
    >
    void Solve(
        const a_T alpha, cref<Tensor1<B_T,B_I>> B,
        const b_T beta,  mref<Tensor1<X_T,X_I>> X_out
    )
    {
        
        if ( B.Dim(0) < RowCount() )
        {
            eprint( ClassName()+"::Solve: B.Dim(0) < RowCount(). Aborting");
            return;
        }
        
        if ( X_out.Dim(0) < RowCount() )
        {
            eprint( ClassName()+"::Solve: X.Dim(0) < RowCount(). Aborting");
            return;
        }
        
        constexpr Int NRHS = 1;
        
        Solve<NRHS,parQ,op>(
            alpha, B.data(),     NRHS,
            beta,  X_out.data(), NRHS
        );
    }


//###########################################################
//####          UpperSolve
//###########################################################

    // Evaluate X = alpha * op(U)^{-1} . B + beta * X
    template<
        Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id,
        typename a_T, typename B_T, typename b_T, typename X_T
    >
    void UpperSolve(
        const a_T alpha, cptr<B_T> B,     const Int ldB,
        const b_T beta,  mptr<X_T> X_out, const Int ldX,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        const std::string tag = ClassName()+"::UpperSolve"
            + "<" + ToString(VarSize)
            + "," + ToString(parQ)
            + "," + ToString(op)
            + "," + TypeName<a_T>
            + "," + TypeName<B_T>
            + "," + TypeName<b_T>
            + "," + TypeName<X_T>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "UpperSolve with Op::Trans not implemented for scalar of complex type."
        );
        TOOLS_PTIMER(timer,tag);
        // No problem if X_out and B overlap, since we load B into X anyways.
        
        if constexpr ( NRHS > VarSize )
        {
            if( nrhs_ != NRHS )
            {
                eprint( tag + ": (NRHS > VarSize) && (nrhs_ != NRHS). Aborting.");
                return;
            }
        }
        
        ReadRightHandSide<NRHS>( B, ldB, nrhs_ );
        
        if( nrhs == ione )
        {
            if( thread_count > Int(1) )
            {
                SN_UpperSolve<false,parQ>();
            }
            else
            {
                SN_UpperSolve<false,Sequential>();
            }
        }
        else
        {
            if( thread_count > Int(1) )
            {
                SN_UpperSolve<true,parQ>();
            }
            else
            {
                SN_UpperSolve<true,Sequential>();
            }
        }

        WriteSolution<NRHS>( alpha, beta, X_out, ldX );
    }

//###########################################################
//####          LowerSolve
//###########################################################

    // Evaluate X = alpha * op(L)^{-1} . B + beta * X
    template<
        Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id,
        typename a_T, typename B_T, typename b_T, typename X_T
    >
    void LowerSolve(
        const a_T alpha, cptr<B_T> B,     const Int ldB,
        const b_T beta,  mptr<X_T> X_out, const Int ldX,
        const Int nrhs_ = ((NRHS > VarSize) ? NRHS : ione )
    )
    {
        const std::string tag = ClassName()+"::LowerSolve"
            + "<" + ToString(VarSize)
            + "," + ToString(parQ)
            + "," + ToString(op)
            + "," + TypeName<a_T>
            + "," + TypeName<B_T>
            + "," + TypeName<b_T>
            + "," + TypeName<X_T>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "LowerSolve with Op::Trans not implemented for scalar of complex type."
        );
        TOOLS_PTIMER(timer,tag);
        
        // No problem if X_out and B overlap, since we load B into X anyways.
        
        if constexpr ( NRHS > VarSize )
        {
            if( nrhs_ != NRHS )
            {
                eprint( tag + ": (NRHS > VarSize) && (nrhs_ != NRHS). Aborting.");
                return;
            }
        }
        
        ReadRightHandSide<NRHS>( B, ldB, nrhs_ );
        
        if( nrhs == ione )
        {
            if( thread_count > Int(1) )
            {
                SN_UpperSolve<false,parQ>();
            }
            else
            {
                SN_UpperSolve<false,Sequential>();
            }
        }
        else
        {
            if( thread_count > Int(1) )
            {
                SN_UpperSolve<true,parQ>();
            }
            else
            {
                SN_UpperSolve<true,Sequential>();
            }
        }

        WriteSolution<NRHS>( alpha, beta, X_out, ldX );
    }

//###########################################################
//#### Supernodal back substitution, both parallel and sequential
//###########################################################

protected:

    // These are the internal solve commands.


    template<bool mult_rhsQ, Parallel_T parQ = Sequential, Op op = Op::Id>
    void SN_Solve()
    {
        static_assert( op == Op::Id, "This feature is currently under developmen" );
        
        // TODO: The only other meaningful (and doable) value for op would be Op::Trans and that only if we use complex numbers. Altogether, this feature seems to be a bit pointless.
        
        if constexpr ( op == Op::Id )
        {
            SN_LowerSolve<mult_rhsQ,parQ>();
            SN_UpperSolve<mult_rhsQ,parQ>();
        }
        // This does not make sense.
//        else if constexpr ( op == Op::ConjTrans )
//        {
//            SN_UpperSolve<mult_rhsQ,parQ>();
//            SN_LowerSolve<mult_rhsQ,parQ>();
//        }
    }

    template<bool mult_rhsQ, Parallel_T parQ = Sequential>
    void SN_UpperSolve()
    {
        // Solves U * X = B and stores the result back into B.
        // Assumes that B has size n x nrhs.
        
        using Solver_T = UpperSolver<mult_rhsQ,Scal,Int,LInt>;
        
        const std::string tag = ClassName()+"::SN_UpperSolve<" + ToString(mult_rhsQ) + "," + ToString(parQ) + "> ( " + ToString(nrhs)+ " )";
        
        TOOLS_PTIMER(timer,tag);
        
        if( !NumericallyFactorizedQ() )
        {
            eprint(tag + ": Nonzero values of matrix have not been passed, yet. Aborting.");
            return;
        }
        
        if( !NumericallyGoodQ() )
        {
            eprint(tag + ": Aborting because numerical factorization was not successful.");
            return;
        }
        
        const Size_T use_threads = ( parQ == Parallel) ? static_cast<Size_T>(thread_count) : Size_T(1);
        
        std::vector<std::unique_ptr<Solver_T>> F_list ( use_threads );
        
        Do<parQ>(
            [&F_list,this]( const Size_T thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            use_threads, use_threads
        );
        
        // Parallel traversal in preorder
        aTree.template Traverse_PreOrdered<parQ>( F_list );
    }

    template<bool mult_rhsQ, Parallel_T parQ = Sequential>
    void SN_LowerSolve()
    {
        // Solves L * X = B and stores the result back into B.
        // Assumes that B has size n x nrhs.
        
        // Use locks if run in parallel.
        using Solver_T = LowerSolver<mult_rhsQ,( parQ == Parallel ? true : false),Scal,Int,LInt>;
        
        const std::string tag = ClassName()+"::SN_LowerSolve<" + ToString(mult_rhsQ) + "," + ToString(parQ) + "> ( " + ToString(nrhs)+ " )";
        
        TOOLS_PTIMER(timer,tag);
        
        if( !NumericallyFactorizedQ() )
        {
            eprint(tag+": Nonzero values of matrix have not been passed, yet. Aborting.");
            return;
        }
        
        if( !NumericallyGoodQ() )
        {
            eprint(tag + ": Aborting because numerical factorization was not successful.");
            return;
        }
        
        const Size_T use_threads = (parQ == Parallel) ? static_cast<Size_T>(thread_count) : Size_T(1);
        
        std::vector<std::unique_ptr<Solver_T>> F_list ( use_threads );
        
        Do<parQ>(
            [&F_list,this]( const Size_T thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            use_threads, use_threads
        );
        
        // Parallel traversal in postorder
        aTree.template Traverse_PostOrdered<parQ>( F_list );
    }
