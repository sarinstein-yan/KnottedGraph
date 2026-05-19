#pragma once

namespace Tools
{
    // TODO: Make the job pointers respect the size of the cache line somehow.
    
    template<IntQ Int_>
    class JobPointers final
    {
    public:
        
        using Int = Int_;
        
        // Default constructor
        JobPointers() = default;
        // Destructor
        ~JobPointers() = default;
        // Copy constructor
        JobPointers( const JobPointers & other ) = default;
        // Copy assignment operator
        JobPointers & operator=( const JobPointers & other ) = default;
        // Move constructor
        JobPointers( JobPointers && other ) = default;
        // Move assignment operator
        JobPointers & operator=( JobPointers && other ) = default;
        
        template<typename I>
        JobPointers( const Int job_count, const I thread_count )
        :   job_ptr ( ToSize_T(thread_count)+1, Int(0) )
        {
            BalanceWorkLoad( job_count );
        }
        
        template<typename T, typename I>
        JobPointers( const Int job_count, cptr<T> costs, const I thread_count, bool accumulateQ = true )
        :   job_ptr (
                std::vector<Int>(
                    static_cast<Size_T>(thread_count+1),
                    Int(0)
                )
            )
        {
            if( accumulateQ )
            {
                BalanceWorkLoad_Accumulated( job_count, costs );
            }
            else
            {
                BalanceWorkLoad( job_count, costs );
            }
        }
        
        template<typename I>
        const Int & operator[]( const I i ) const
        {
            return job_ptr[ToSize_T(i)];
        }
        
        template<typename I>
        const Int & operator()( const I i ) const
        {
            return job_ptr[ToSize_T(i)];
        }
        
        Int Size() const
        {
            return static_cast<Int>(job_ptr.size());
        }
        
        Int size() const
        {
            return static_cast<Int>(job_ptr.size());
        }
        
        Int ThreadCount() const
        {
            if( job_ptr.size() <= static_cast<Size_T>(0) )
            {
                wprint("ThreadCount: empty JobPointers detected.");
            }
            return ( job_ptr.size() <= static_cast<Size_T>(1) ? Int(1) : static_cast<Int>(job_ptr.size()-1) );
        }
        
        Int JobCount() const
        {
            return static_cast<Int>(job_ptr.back());
        }
        
        const Int * data() const
        {
            return job_ptr.data();
        }
        
        const std::vector<Int> & Vector() const
        {
            return job_ptr;
        }
        
        auto Begin() const
        {
            return job_ptr.begin();
        }
        
        auto End() const
        {
            return job_ptr.end();
        }
        
        void BalanceWorkLoad( const Int job_count )
        {
            const Int thread_count = ThreadCount();
            
            for( Int k = 0; k < thread_count+1; ++k )
            {
                job_ptr[k] = JobPointer( job_count, thread_count, k );
            }
        }
        
        template<typename T>
        void BalanceWorkLoad_Accumulated( const Int job_count, cptr<T> costs )
        {
            const Int thread_count = ThreadCount();
            
            // This function reads in a list job_acc_costs of accumulated costs, then allocates job_ptr as a vector of size thread_count + 1, and writes the work distribution to it.
            // Assigns threads to consecutive chunks jobs, ..., job_ptr[k+1]-1 of jobs.
            // Uses a binary search to find the chunk boundaries.
            // The cost of the i-th job is job_acc_costs[i+1] - job_acc_costs[i].
            // The cost of the k-th thread goes from job no job_ptr[k] to job no job_ptr[k+1] (as always in C/C++, job_ptr[k+1] points _after_ the last job.
        
            
//            TOOLS_PTIMER(timer,MethodName("BalanceWorkLoad_Accumulated"));
            
            T * acc_costs = nullptr;
            
            safe_alloc( acc_costs, static_cast<Size_T>(job_count + Int(1)) );
            
            acc_costs[0] = T(0);
            // Is parallel accumulation really a good idea here?
            Accumulate<Parallel>( &costs[0], &acc_costs[1], job_count, thread_count );

            BalanceWorkLoad( job_count, acc_costs);
            
            safe_free( acc_costs );
        }
        
        template<typename T>
        void BalanceWorkLoad( const Int job_count, cptr<T> acc_costs )
        {
            const Int thread_count = ThreadCount();
            
            // This function reads in a list job_acc_costs of accumulated costs, then allocates job_ptr as a vector of size thread_count + 1, and writes the work distribution to it.
            // Assigns threads to consecutive chunks jobs, ..., job_ptr[k+1]-1 of jobs.
            // Uses a binary search to find the chunk boundaries.
            // The cost of the i-th job is job_acc_costs[i+1] - job_acc_costs[i].
            // The cost of the k-th thread goes from job no job_ptr[k] to job no job_ptr[k+1] (as always in C/C++, job_ptr[k+1] points _after_ the last job.
            
//            TOOLS_PTIMER(timer,MethodName("BalanceWorkLoad"));

            job_ptr[static_cast<Size_T>(thread_count)] = job_count;

            const Int naive_chunk_size = ((job_count + thread_count) - Int(1)) / thread_count;
            
            const T total_cost = acc_costs[job_count];
            
            if( total_cost <= T(0) )
            {
                wprint(MethodName("BalanceWorkLoad") + ": Total cost is <= 0.");
                
                TOOLS_DDUMP(job_count);
                TOOLS_DDUMP(thread_count);
                
                logvalprint( "acc_costs", OutString::FromVector(acc_costs, job_count + Int(1)) );
                
                std::fill( job_ptr.begin(), job_ptr.end(), Int(0));
                            
                return;
            }
            
            const T per_thread_cost = (total_cost + thread_count - Int(1)) / thread_count;

            // binary search for best work load distribution
            
            //  There is quite a lot false sharing in this loop... so better not parallelize it.
            for( Int thread = 0; thread < thread_count - Int(1); ++thread)
            {
                // each thread (other than the last one) is required to have at least this accumulated cost
                T target = Min( total_cost, static_cast<T>(per_thread_cost * (thread + Int(1))) );
                
                Int pos;
                // find an index a such that b_row_acc_costs[ a ] < target;
                // taking naive_chunk_size * thread as initial guess, because that might be nearly correct for costs that are evenly distributed over the block rows
                pos = thread + Int(1);
                Int a = Min(job_count, static_cast<Int>(naive_chunk_size * pos) );
                while( acc_costs[a] >= target )
                {
                    --pos;
                    a = Min(job_count, static_cast<Int>(naive_chunk_size * pos) );
                };
                
                // find an index  b such that b_row_acc_costs[ b ] >= target;
                // taking naive_chunk_size * (thread + 1) as initial guess, because that might be nearly correct for costs that are evenly distributed over the block rows
                pos = thread + Int(1);
                Int b = Min(job_count, static_cast<Int>(naive_chunk_size * pos) );
                while( (b < job_count) && (acc_costs[b] < target) )
                {
                    ++pos;
                    b = Min(job_count, static_cast<Int>(naive_chunk_size * pos) );
                };
                
                // binary search
                while( b > a + Int(1) )
                {
                    const Int c = a + (b-a)/2;
                    if( acc_costs[c] > target )
                    {
                        b = c;
                    }
                    else
                    {
                        a = c;
                    }
                }

                job_ptr[static_cast<Size_T>(thread + 1)] = b;
            }   
        }
        
        template<typename I>
        void LoadFromBuffer( cptr<I> buffer )
        {
            std::copy_n( buffer, ThreadCount() + 1, &job_ptr[0] );
        }
        

        [[nodiscard]] std::string friend ToString( const JobPointers & J)
        {
            return OutString::FromVector( &J.job_ptr[0], J.Size() );
        }
        
    private:
        
        std::vector<Int> job_ptr;
        
        
    public:
        
        static std::string MethodName( const std::string & tag )
        {
            return ClassName() + "::" + tag;
        }
        
        static std::string ClassName()
        {
            return std::string("JobPointers") + "<" + TypeName<Int> + ">";
        }
        
        
    }; // JobPointers
    
} // namespace Tools
