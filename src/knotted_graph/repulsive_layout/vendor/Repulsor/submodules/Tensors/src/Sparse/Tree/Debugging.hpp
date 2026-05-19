public:

    class DebugWorker
    {
    public:
     
        cref<Tree> tree = nullptr;
        
        DebugWorker( const Tree & tree_ )
        :   tree( tree_ )
        {}
        
        void operator()( const Int node )
        {
            tree.IncrementChecklist( node );
        }
        
        std::string ClassName()
        {
            return "DebugWorker";
        }
    };

    void AllocateCheckList() const
    {
//        print(ClassName()+"::AllocateCheckList");
        check_list = Tensor1<Int,Int> ( n, 0 );
    }

    void IncrementChecklist( const Int node ) const
    {
        if( node >= check_list.Size() )
        {
            eprint(ClassName()+"::IncrementChecklist: check_list is too short (size  = " + ToString(check_list.Size()) + ").");
        }
        else
        {
            ++check_list[node];
        }
    }

    bool PrintCheckList() const
    {
        std::vector<Int> not_visited;
        std::vector<Int> multiply_visited;
        for( Int node = 0; node < check_list.Size() - Int(1); ++node )
        {
            if( check_list[node] < Int(1) )
            {
                not_visited.push_back(node);
            }
            
            if( check_list[node] > Int(1) )
            {
                multiply_visited.push_back(node);
            }
        }
        
        if( not_visited.size() > Size_T(0) )
        {
            eprint(ClassName()+"::PrintCheckList: There are are " + ToString(not_visited.size()) + " unvisited nodes.");
            
            TOOLS_DUMP(not_visited);
        }
        
        if( multiply_visited.size() > Size_T(0) )
        {
            eprint(ClassName()+"::PrintCheckList: There are are " + ToString(multiply_visited.size()) + " multiply visited nodes.");
            
            TOOLS_DUMP(multiply_visited);
        }
        
        return (not_visited.size() == Size_T(0)) && (multiply_visited.size() == Size_T(0));
    }

    Tensor1<Int,Int> & CheckList() const
    {
        return check_list;
    }






    bool Traverse_PostOrdered_Test() const
    {
        TOOLS_PTIMER(timer,ClassName()+"::Traverse_PostOrdered_Test");
        AllocateCheckList();

        std::vector<std::unique_ptr<DebugWorker>> workers ( static_cast<Size_T>(thread_count) );
        
        Do<parQ>(
            [this,&workers]( const Size_T thread )
            {
                workers[thread] = std::make_unique<DebugWorker>( *this );
            },
            static_cast<Size_T>(thread_count), static_cast<Size_T>(thread_count)
        );
        
        Traverse_PostOrdered( workers );
        
        bool succeededQ = PrintCheckList();
        
        if( succeededQ )
        {
            print(ClassName()+"::Traverse_PostOrdered_Test succeeded.");
            logprint(ClassName()+"::Traverse_PostOrdered_Test succeeded.");
        }
        else
        {
            eprint(ClassName()+"::Traverse_PostOrdered_Test failed.");
        }

        return succeededQ;
    }



    bool Traverse_PreOrdered_Test() const
    {
        TOOLS_PTIMER(timer,ClassName()+"::Traverse_PreOrdered_Test");
        AllocateCheckList();

        std::vector<std::unique_ptr<DebugWorker>> workers ( static_cast<Size_T>(thread_count) );
        
        Do<parQ>(
            [this,&workers]( const Size_T thread )
            {
                workers[thread] = std::make_unique<DebugWorker>( *this );
            },
            static_cast<Size_T>(thread_count), static_cast<Size_T>(thread_count)
        );
        
        Traverse_PreOrdered( workers );
        
        bool succeededQ = PrintCheckList();
        
        if( succeededQ )
        {
            print(ClassName()+"::Traverse_PreOrdered_Test succeeded.");
            logprint(ClassName()+"::Traverse_PreOrdered_Test succeeded.");
        }
        else
        {
            eprint(ClassName()+"::Traverse_PreOrdered_Test failed.");
        }
        
        return succeededQ;
    }
