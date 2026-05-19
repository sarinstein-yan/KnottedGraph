public:

    template<typename ExtScal>
    int NumericFactorization(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        switch( factorization_method )
        {
            case FactorizationMethod_T::Multifrontal:
            {
                return NumericFactorization_Multifrontal( A_val_, reg_ );
            }
            case FactorizationMethod_T::LeftLooking:
            {
                return NumericFactorization_LeftLooking ( A_val_, reg_ );
            }
            default:
            {
                return NumericFactorization_Multifrontal( A_val_, reg_ );
            }
        }
    }

    template<typename ExtScal>
    int NumericFactorization_Multifrontal(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_Multifrontal<" + TypeName<ExtScal> + ">";
    
        TOOLS_PTIMER(timer,tag);
        
        SymbolicFactorization();
        ReadNonzeroValues( A_val_, reg_ );
        ClearFactors();
        
        // TODO: We might want to use memory stacks or pools
        // TODO: if we don't want to rely on the sytem's way to handle the memory.
        // TODO: Each thread should probably get its own memory stack.
        // TODO: This stack is a expandable container.
        // TODO: Supernodes are visited in depth-first order.
        // TODO: When visited for the first time,
        // TODO: the supernode's update block is pushed to stack.
        // TODO: After a supernode has fetched the updates from its children,
        // TODO: the childrens' memory is popped.
        // TODO: Children need to know: A pointer to their block and maybe also the thread
        // TODO: on which the block was allocated.
        // TODO: Both might be fused into a simple struct.
        
        SN_updates = std::vector<Update_T> ( ToSize_T(SN_count) );

        Size_T use_threads = ToSize_T(thread_count);
        
        std::vector<std::unique_ptr<Factorizer_MF_T>> SN_list (use_threads);
        
        Do<parQ>(
            [&SN_list,this]( const Size_T thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_MF_T>(*this);
            },
            use_threads, use_threads
        );
        
        Factorizer_MF_T worker(*this);
        
        // (Parallel) traversal in postorder
        if( thread_count > Int(1) )
        {
            aTree.template Traverse_PostOrdered<parQ>( SN_list );
        }
        else
        {
            aTree.template Traverse_PostOrdered<Sequential>( SN_list );
        }
        
        SN_data.goodQ = true;
        
        Do<parQ>(
            [&SN_list,this]( const Size_T thread )
            {
                SN_data.goodQ = SN_data.goodQ && SN_list[thread]->GoodQ();
            },
            use_threads, use_threads
        );
        
        if( !SN_data.goodQ )
        {
            eprint(ClassName()+"::NumericFactorization_Multifrontal: Could not complete numeric factorization. Matrix is not (sufficiently) positive-definite.");
        }
        
        // We mark this as factorized in any case to not attempt the factorization again.
        SN_data.factorizedQ = true;
        
        SN_updates = std::vector<Update_T>();
        
        return (SN_data.goodQ ? 0 : 1);
    }


    template<typename ExtScal>
    int NumericFactorization_LeftLooking(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_LeftLooking<" + TypeName<ExtScal> + ">";
       
        TOOLS_PTIMER(timer,tag);
        
        SymbolicFactorization();

        ReadNonzeroValues( A_val_, reg_ );
        
        ClearFactors();
        
        Size_T use_threads = ToSize_T(thread_count);
        
        std::vector<std::unique_ptr<Factorizer_LL_T>> SN_list (use_threads);
        
        Do<parQ>(
            [&SN_list,this]( const Size_T thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_LL_T>(*this);
            },
            use_threads, use_threads
        );
        
        // (Parallel) traversal in postorder
        if( thread_count > Int(1) )
        {
            aTree.template Traverse_PostOrdered<parQ>( SN_list );
        }
        else
        {
            aTree.template Traverse_PostOrdered<Sequential>( SN_list );
        }
        
        SN_data.goodQ = true;
        
        Do<parQ>(
            [&SN_list,this]( const Size_T thread )
            {
                SN_data.goodQ = SN_data.goodQ && SN_list[thread]->GoodQ();
            },
            use_threads, use_threads
        );
        
        if( !SN_data.goodQ )
        {
            eprint(ClassName()+"::NumericFactorization_LeftLooking: Could not complete numeric factorization. Matrix is not (sufficiently) positive-definite.");
        }
        
        // We mark this as factorized in any case to not attempt the factorization again.
        SN_data.factorizedQ = true;
        
        return (SN_data.goodQ ? 0 : 1);
    }



    template<typename ExtScal>
    void ReadNonzeroValues( cptr<ExtScal> A_val_, const ExtScal reg_ )
    {
        TOOLS_PTIMER(timer,ClassName()+"::ReadNonzeroValues<" + TypeName<ExtScal> + ">");
        
        Do<parQ>(
            [&]( const LInt i )
            {
                A_val[i] = static_cast<Scal>(A_val_[A_inner_perm[i]]);
            },
            A_inner_perm.Size(), static_cast<LInt>(thread_count)
        );
        
        reg = reg_;
        
        this->ClearCache();
    }


    bool NumericallyFactorizedQ() const
    {
        return SN_data.factorizedQ;
    }

    bool NumericallyGoodQ() const
    {
        return SN_data.goodQ;
    }

    LInt RequiredTriangularSize() const
    {
        if( SN_initializedQ )
        {
            return SN_tri_ptr[SN_count];
        }
        else
        {
            return 0;
        }
    }

    LInt RequiredRectangularSize() const
    {
        if( SN_initializedQ )
        {
            return SN_rec_ptr[SN_count];
        }
        else
        {
            return 0;
        }
    }

    LInt CurrentTriangularSize() const
    {
        return SN_data.tri_val.Size();
    }

    LInt CurrentRectangularSize() const
    {
        return SN_data.rec_val.Size();
    }


    void SwapNumericalFactorization( NumericalFactorization_T & cont ) noexcept
    {
        using std::swap;
        
        swap( this->SN_data, cont );
        
        if( SN_data.factorizedQ && !SN_data.goodQ )
        {
            wprint(MethodName("SwapNumericalFactorization") + ": Loaded numerical factorizations states it were factorized, but incorrect. Treating it as unfactorized.");
            
            SN_data.factorizedQ = false;
        }
        
        if( SN_data.factorizedQ )
        {
            if(
                (RequiredTriangularSize() > CurrentTriangularSize())
                ||
                (RequiredRectangularSize() > CurrentRectangularSize())
            )
            {
                wprint(MethodName("SwapNumericalFactorization") + ": Loaded numerical factorizations states it were factorized, but its size does not match the symbolic factorization. Treating it as unfactorized.");
                
                SN_data.factorizedQ = false;
            }
        }
    }

    void ClearFactors()
    {
        TOOLS_PTIMER(timer,ClassName()+"::ClearFactors");
        
        SN_data.tri_val.template RequireSize<false>( RequiredTriangularSize()  );
        SN_data.rec_val.template RequireSize<false>( RequiredRectangularSize() );
        
        SN_data.tri_val.SetZero( thread_count );
        SN_data.rec_val.SetZero( thread_count );
        
        SN_data.factorizedQ = false;
        SN_data.goodQ       = false;
    }
