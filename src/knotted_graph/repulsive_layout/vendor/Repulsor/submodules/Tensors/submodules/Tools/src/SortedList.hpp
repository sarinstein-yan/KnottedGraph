
namespace Tools
{
    template<typename T, SignedIntQ Int>    // TODO: Why signed integer?
    class SortedList final
    {
        std::vector<T> list;
        
        // TODO: Replace by
//        HeapArray<T,Int> list;
        
    public:
        
        mptr<T> begin()
        {
            return &*list.begin();
        }
        
        cptr<T> begin() const
        {
            return &*list.cbegin();
        }
        
        mptr<T> end()
        {
            return &*list.end();
        }
        
        cptr<T> end() const
        {
            return &*list.cend();
        }
        
        // Default constructor
        SortedList() = default;
        // Destructor
        ~SortedList() = default;
        // Copy constructor
        SortedList( const SortedList & other ) = default;
        // Copy assignment operator
        SortedList & operator=( const SortedList & other ) = default;
        // Move constructor
        SortedList( SortedList && other ) = default;
        // Move assignment operator
        SortedList & operator=( SortedList && other ) = default;
        
        explicit SortedList( const Int n )
        {
            Reserve(n);
        }
        
        void Clear()
        {
            list.clear();
        }
        
        void Reserve( const Size_T n )
        {
            list.reserve(n);
        }


        Int FindPosition( cref<T> X ) const
        {
            //Finds first position whose element is not smaller than X.

            return static_cast<Int>(
                std::distance(
                    list.begin(),
                    std::lower_bound(list.begin(), list.end(), X)
                )
            );
        }

        
        Int Find( cref<T> X ) const
        {
            if( Size() == 0 )
            {
                return -1;
            }
            
            Int i = FindPosition(X);

            return (list[i] == X) ? i : -1;
        }
        
        bool Insert( cref<T> X )
        {
            Int i = FindPosition(X);
            
            if( i == Size() )
            {
                list.push_back(X);
                return true;
            }
            
            if( list[i] != X )
            {
                list.insert( list.begin()+i, X );
                return true;
            }
            else
            {
                return false;
            }
            
        }
        
        bool Push( cref<T> X )
        {
            return Insert(X);
        }
        
        bool Drop( cref<T> X )
        {
            Int i = Find(X);
            if( i >= Int(0) )
            {
                list.erase(list.begin()+i);
                return true;
            }
            else
            {
                return false;
            }
        }
        
        T Pop()
        {
            const T r = list.back();
            
            list.pop_back();
            
            return r;
        }
        
        Int Size() const
        {
            return static_cast<Int>(list.size());
        }
        
        T Min() const
        {
            return list.front();
        }
        
        T Max() const
        {
            return list.back();
        }
        
        const T & operator[]( const Int i ) const
        {
            return list[i];
        }
        
        friend bool IntersectingQ(
            cref<SortedList<T,Int>> a,
            cref<SortedList<T,Int>> b
        )
        {
            if( (a.Max() < b.Min()) || (b.Max() < a.Min()) )
            {
                return false;
            }
            else
            {
                //https://www.geeksforgeeks.org/union-and-intersection-of-two-sorted-arrays-2/
                Int i = 0;
                Int j = 0;
                const Int m = a.Size();
                const Int n = b.Size();
                
                
                while (i < m && j < n)
                {
                    if (a[i] < b[j])
                    {
                        i++;
                    }
                    else
                    {
                        if (b[j] < a[i])
                        {
                            j++;
                        }
                        else /* if a[i] == b[j] */
                        {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
        
        friend void Intersection(
            cref<SortedList<T,Int>> a,
            cref<SortedList<T,Int>> b,
            mref<SortedList<T,Int>> c
        )
        {
            c.Clear();
            
            //https://www.geeksforgeeks.org/union-and-intersection-of-two-sorted-arrays-2/

            const Int m = a.Size();
            const Int n = b.Size();
            
            Int i = 0;
            Int j = 0;
            
            while (i < m && j < n)
            {
                const T a_i = a[i];
                const T b_j = b[j];
                
                if (a_i < b_j)
                {
                    ++i;
                }
                else
                {
                    if (b_j < a_i)
                    {
                        j++;
                    }
                    else /* if a[i] == b[j] */
                    {
                        c.Insert(a_i);
                        ++i;
                        ++j;
                        
                    }
                }
            }
        }
        
        friend bool operator==(
            cref<SortedList<T,Int>> a,
            cref<SortedList<T,Int>> b
        )
        {
            if( a.Size() != b.Size() )
            {
                return false;
            }
            
            const Int n = a.Size();
            for( Int i = 0; i < n; ++i )
            {
                if( a[i] != b[i] )
                {
                    return false;
                }
            }
            
            return true;
        }
        
        friend bool operator!=(
            cref<SortedList<T,Int>> a,
            cref<SortedList<T,Int>> b
        )
        {
            if( a.Size() != b.Size() )
            {
                return true;
            }
            
            const Int n = a.Size();
            for( Int i = 0; i < n; ++i )
            {
                if( a[i] != b[i] )
                {
                    return true;
                }
            }
            
            return false;
        }
        
        [[nodiscard]] friend std::string ToString( cref<SortedList> L )
        {
            return OutString::FromVector( &L.list[0], L.list.size() );
        }
        
    };

} //namespace Tools
