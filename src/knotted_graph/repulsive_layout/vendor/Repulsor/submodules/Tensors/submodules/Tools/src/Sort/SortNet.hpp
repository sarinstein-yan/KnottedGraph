#pragma once

#include <functional>

namespace Tools
{
   
    template<bool reverseQ = false, IntQ T, typename C = std::less<T>>
    TOOLS_FORCE_INLINE void CompSwap( mref<T> a, mref<T> b, C && comp = C() )
    {
        if constexpr ( reverseQ )
        {
            std::tie(b,a) = std::minmax({a,b},std::forward<C>(comp));
        }
        else
        {
            std::tie(a,b) = std::minmax({a,b},std::forward<C>(comp));
        }
    }
    
    template<bool reverseQ = false, std::floating_point T, typename C = std::less<T>>
    TOOLS_FORCE_INLINE void CompSwap( mref<T> a, mref<T> b, C && comp = C()  )
    {
        constexpr bool lessQ = SameQ<C,std::less<T>> || SameQ<C,std::less_equal<T>>;
        
        constexpr bool greaterQ = SameQ<C,std::greater<T>> || SameQ<C,std::greater_equal<T>>;
        
        if constexpr ( lessQ || greaterQ )
        {
            if constexpr ( lessQ == reverseQ )
            {
                std::tie(b,a) = MinMax(a,b);
            }
            else
            {
                std::tie(a,b) = MinMax(a,b);
            }
        }
        else
        {
            if constexpr ( reverseQ )
            {
                std::tie(b,a) = std::minmax({a,b},std::forward<C>(comp));
            }
            else
            {
                std::tie(a,b) = std::minmax({a,b},std::forward<C>(comp));
            }
        }
    }
    
    template<bool reverseQ = false, typename T, typename C = std::less<T>>
    TOOLS_FORCE_INLINE void CompSwap( mref<T> a, mref<T> b, C && comp = C() )
    {
        static_assert(!std::integral<T>,"");
        static_assert(!std::floating_point<T>,"");
        
        // For generic objects we use std::swap (or customized swap routines) and rely on move semantics.
        // There will probably be a branch in the code, but that is well worth it for bigger data structures.
        if constexpr ( reverseQ )
        {
            if( comp(a,b) )
            {
                using std::swap;
                swap(a,b);
            }
        }
        else
        {
            if( comp(b,a) )
            {
                using std::swap;
                swap(b,a);
            }
        }
    }
    
    
    template<int n, bool reverseQ = false>
    struct SortNet
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer( a, b, n );
            this->operator()( b, std::forward<C>(comp) );
        }
        
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            std::sort( a, a+n, std::forward<C>(comp) );
        }
    };

    template<bool reverseQ>
    struct SortNet<1,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<1>( a, b );
            this->operator()( b, std::forward<C>(comp) );
        }
     
        template<typename T,typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            (void)a;
            (void)comp;
        }
    };
    
    template<bool reverseQ>
    struct SortNet<2,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<2>( a, b );
            this->operator()( b, std::forward<C>(comp) );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
        }
    };
    
    template<bool reverseQ>
    struct SortNet<3,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<3>( a, b );
            this->operator()( b, std::forward<C>(comp) );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
            s(a[0],a[2]);
            s(a[1],a[2]);
        }
    };
    
    template<bool reverseQ>
    struct SortNet<4,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<4>( a, b );
            this->operator()( b, std::forward<C>(comp) );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            
            s(a[0],a[2]);
            s(a[1],a[3]);
            
            s(a[1],a[2]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<5,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<5>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            
            s(a[1],a[3]);
            s(a[2],a[4]);
            
            s(a[0],a[2]);
            s(a[1],a[4]);
            
            s(a[1],a[2]);
            s(a[3],a[4]);
            
            s(a[2],a[3]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<6,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<6>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            
            s(a[0],a[2]);
            s(a[3],a[5]);
            s(a[1],a[4]);
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            
            s(a[1],a[2]);
            s(a[3],a[4]);

            s(a[2],a[3]);
        }
    };
    
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<7,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<7>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[1],a[2]);
            s(a[3],a[4]);
            s(a[5],a[6]);

            s(a[0],a[2]);
            s(a[4],a[6]);
            s(a[3],a[5]);

            s(a[2],a[6]);
            s(a[1],a[5]);
            s(a[0],a[4]);

            s(a[2],a[5]);
            s(a[0],a[3]);

            s(a[2],a[4]);
            s(a[1],a[3]);

            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<8,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<8>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[2]);
            s(a[1],a[3]);
            s(a[4],a[6]);
            s(a[5],a[7]);
            
            s(a[0],a[4]);
            s(a[1],a[5]);
            s(a[2],a[6]);
            s(a[3],a[7]);
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            s(a[6],a[7]);
            
            s(a[2],a[4]);
            s(a[3],a[5]);
            
            s(a[1],a[4]);
            s(a[3],a[6]);
            
            s(a[1],a[2]);
            s(a[3],a[4]);
            s(a[5],a[6]);
            
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<9,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<9>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[1],a[8]);
            s(a[2],a[7]);
            s(a[3],a[6]);
            s(a[4],a[5]);
            
            s(a[1],a[4]);
            s(a[0],a[2]);
            s(a[5],a[8]);
            s(a[6],a[7]);
            
            s(a[2],a[6]);
            s(a[7],a[8]);
            s(a[0],a[3]);
            s(a[4],a[5]);
            
            s(a[0],a[1]);
            s(a[3],a[5]);
            s(a[6],a[7]);
            s(a[2],a[4]);
            
            s(a[1],a[3]);
            s(a[5],a[7]);
            s(a[4],a[6]);
            
            s(a[1],a[2]);
            s(a[3],a[4]);
            s(a[5],a[6]);
            s(a[7],a[8]);
            
            s(a[2],a[3]);
            s(a[4],a[5]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<10,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<10>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            s(a[6],a[7]);
            s(a[8],a[9]);
            
            s(a[4],a[9]);
            s(a[0],a[5]);
            s(a[1],a[8]);
            s(a[3],a[7]);
            s(a[2],a[6]);
            
            s(a[0],a[2]);
            s(a[3],a[6]);
            s(a[7],a[9]);
            s(a[1],a[4]);
            s(a[5],a[8]);
            
            s(a[0],a[1]);
            s(a[2],a[7]);
            s(a[8],a[9]);
            s(a[4],a[6]);
            s(a[3],a[5]);
            
            s(a[2],a[4]);
            s(a[6],a[8]);
            s(a[1],a[3]);
            s(a[5],a[7]);
            
            s(a[1],a[2]);
            s(a[3],a[4]);
            s(a[5],a[6]);
            s(a[7],a[8]);
            
            s(a[2],a[3]);
            s(a[4],a[5]);
            s(a[6],a[7]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<11,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<11>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[ 0],a[ 9]);
            s(a[ 1],a[ 8]);
            s(a[ 2],a[ 7]);
            s(a[ 3],a[ 6]);
            s(a[ 4],a[ 5]);
            
            s(a[ 0],a[ 3]);
            s(a[ 1],a[ 2]);
            s(a[ 4],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            
            s(a[ 0],a[ 1]);
            s(a[ 2],a[ 3]);
            s(a[ 5],a[ 8]);
            s(a[ 6],a[ 7]);
            s(a[ 9],a[10]);
            
            s(a[ 1],a[ 2]);
            s(a[ 4],a[ 6]);
            s(a[ 8],a[10]);
            s(a[ 5],a[ 9]);
            
            s(a[ 0],a[ 4]);
            s(a[ 7],a[ 8]);
            s(a[ 1],a[ 5]);
            s(a[ 2],a[ 9]);
            s(a[ 3],a[ 6]);
            
            s(a[ 1],a[ 4]);
            s(a[ 2],a[ 3]);
            s(a[ 5],a[ 7]);
            s(a[ 6],a[ 9]);
            
            s(a[ 2],a[ 4]);
            s(a[ 6],a[ 7]);
            s(a[ 8],a[ 9]);
            s(a[ 3],a[ 5]);
            
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 6]);
            s(a[ 7],a[ 8]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<12,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<12>( a, b );
            this->operator()( b, comp );
        }
     
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[ 0],a[ 6]);
            s(a[ 1],a[ 7]);
            s(a[ 2],a[ 8]);
            s(a[ 3],a[ 9]);
            s(a[ 4],a[10]);
            s(a[ 5],a[11]);
            
            s(a[ 0],a[ 3]);
            s(a[ 1],a[ 4]);
            s(a[ 2],a[ 5]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[10]);
            s(a[ 8],a[11]);
            
            s(a[ 0],a[ 1]);
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 8]);
            s(a[ 6],a[ 7]);
            s(a[10],a[11]);
            
            s(a[ 1],a[ 2]);
            s(a[ 3],a[ 6]);
            s(a[ 4],a[ 5]);
            s(a[ 7],a[ 8]);
            s(a[ 9],a[10]);
            
            s(a[ 0],a[ 1]);
            s(a[ 2],a[ 9]);
            s(a[10],a[11]);
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 8]);
            s(a[ 6],a[ 7]);
            
            s(a[ 1],a[ 3]);
            s(a[ 4],a[ 7]);
            s(a[ 8],a[10]);
            s(a[ 2],a[ 6]);
            s(a[ 5],a[ 9]);
            
            s(a[ 2],a[ 3]);
            s(a[ 4],a[ 6]);
            s(a[ 8],a[ 9]);
            s(a[ 5],a[ 7]);
            
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 6]);
            s(a[ 7],a[ 8]);
        }
    };
   
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<13,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<13>( a, b );
            this->operator()( b, comp );
        }
     
        // This should be a depth-optimal solution.
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[ 3],a[12]);
            s(a[ 4],a[11]);
            s(a[ 5],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            
            s(a[ 0],a[ 7]);
            s(a[ 1],a[ 6]);
            s(a[ 2],a[ 5]);
            s(a[ 3],a[ 4]);
            s(a[11],a[12]);
            
            s(a[ 0],a[ 3]);
            s(a[ 1],a[ 2]);
            s(a[ 4],a[ 7]);
            s(a[ 5],a[ 6]);
            s(a[ 8],a[11]);
            s(a[ 9],a[10]);
            
            s(a[ 0],a[ 1]);
            s(a[ 2],a[ 8]);
            s(a[10],a[11]);
            s(a[ 3],a[ 9]);
            s(a[ 4],a[ 5]);
            s(a[ 6],a[12]);
            
            s(a[ 1],a[ 4]);
            s(a[ 2],a[ 3]);
            s(a[ 5],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            
            s(a[ 1],a[ 2]);
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 8]);
            s(a[11],a[12]);
            s(a[ 6],a[10]);
            s(a[ 7],a[ 9]);
            
            s(a[ 2],a[ 3]);
            s(a[ 4],a[ 6]);
            s(a[ 9],a[11]);
            s(a[ 5],a[ 7]);
            s(a[ 8],a[10]);
            
            s(a[ 4],a[ 5]);
            s(a[ 6],a[ 7]);
            s(a[ 8],a[ 9]);
            s(a[10],a[11]);
            
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 6]);
            s(a[ 7],a[ 8]);
            s(a[ 9],a[10]);
            s(a[11],a[12]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<14,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<14>( a, b );
            this->operator()( b, comp );
        }
     
        // This should be a depth-optimal solution.
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[ 2],a[13]);
            s(a[ 3],a[12]);
            s(a[ 4],a[11]);
            s(a[ 5],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            
            s(a[ 0],a[ 7]);
            s(a[ 1],a[ 6]);
            s(a[ 2],a[ 5]);
            s(a[ 3],a[ 4]);
            s(a[10],a[13]);
            s(a[11],a[12]);
            
            s(a[ 0],a[ 3]);
            s(a[ 1],a[ 2]);
            s(a[ 4],a[ 7]);
            s(a[ 5],a[ 6]);
            s(a[ 8],a[11]);
            s(a[ 9],a[10]);
            
            s(a[ 0],a[ 1]);
            s(a[ 2],a[ 8]);
            s(a[10],a[11]);
            s(a[ 3],a[ 9]);
            s(a[ 4],a[ 5]);
            s(a[ 6],a[12]);
            s(a[ 7],a[13]);
            
            s(a[ 1],a[ 4]);
            s(a[ 2],a[ 3]);
            s(a[ 5],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            s(a[12],a[13]);
            
            s(a[ 1],a[ 2]);
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 8]);
            s(a[11],a[12]);
            s(a[ 6],a[10]);
            s(a[ 7],a[ 9]);
            
            s(a[ 2],a[ 3]);
            s(a[ 4],a[ 6]);
            s(a[ 9],a[11]);
            s(a[12],a[13]);
            s(a[ 5],a[ 7]);
            s(a[ 8],a[10]);
            
            s(a[ 4],a[ 5]);
            s(a[ 6],a[ 7]);
            s(a[ 8],a[ 9]);
            s(a[10],a[11]);
            
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 6]);
            s(a[ 7],a[ 8]);
            s(a[ 9],a[10]);
            s(a[11],a[12]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<15,reverseQ>
    
    {   template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<15>( a, b );
            this->operator()( b, comp );
        }
     
        // This should be a depth-optimal solution.
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[ 1],a[14]);
            s(a[ 2],a[13]);
            s(a[ 3],a[12]);
            s(a[ 4],a[11]);
            s(a[ 5],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            
            s(a[ 0],a[ 7]);
            s(a[ 1],a[ 6]);
            s(a[ 2],a[ 5]);
            s(a[ 3],a[ 4]);
            
            s(a[ 9],a[14]);
            s(a[10],a[13]);
            s(a[11],a[12]);
            
            s(a[ 0],a[ 3]);
            s(a[ 1],a[ 2]);
            s(a[ 4],a[ 7]);
            s(a[ 5],a[ 6]);
            s(a[ 8],a[11]);
            s(a[ 9],a[10]);
            s(a[13],a[14]);
            
            s(a[ 0],a[ 1]);
            s(a[ 2],a[ 8]);
            s(a[10],a[11]);
            s(a[ 3],a[ 9]);
            s(a[ 4],a[ 5]);
            s(a[ 6],a[12]);
            s(a[ 7],a[13]);
            
            
            s(a[ 1],a[ 4]);
            s(a[ 2],a[ 3]);
            s(a[ 5],a[10]);
            s(a[ 6],a[ 9]);
            s(a[ 7],a[ 8]);
            s(a[11],a[14]);
            s(a[12],a[13]);
            
            s(a[ 1],a[ 2]);
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 8]);
            s(a[11],a[12]);
            s(a[13],a[14]);
            s(a[ 6],a[10]);
            s(a[ 7],a[ 9]);
            
            s(a[ 2],a[ 3]);
            s(a[ 4],a[ 6]);
            s(a[ 9],a[11]);
            s(a[12],a[13]);
            s(a[ 5],a[ 7]);
            s(a[ 8],a[10]);
            
            s(a[ 4],a[ 5]);
            s(a[ 6],a[ 7]);
            s(a[ 8],a[ 9]);
            s(a[10],a[11]);
            
            s(a[ 3],a[ 4]);
            s(a[ 5],a[ 6]);
            s(a[ 7],a[ 8]);
            s(a[ 9],a[10]);
            s(a[11],a[12]);
        }
    };
    
    // https://www.angelfire.com/blog/ronz/Articles/999SortingNetworksReferen.html
    template<bool reverseQ>
    struct SortNet<16,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<16>( a, b );
            this->operator()( b, comp );
        }
        
        // This should be a depth-optimal solution.
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            // https://bertdobbelaere.github.io/sorting_networks.html#N32L185D14
            
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[13]);
            s(a[1],a[12]);
            s(a[2],a[15]);
            s(a[3],a[14]);
            s(a[4],a[8]);
            s(a[5],a[6]);
            s(a[7],a[11]);
            s(a[9],a[10]);

            s(a[0],a[5]);
            s(a[1],a[7]);
            s(a[2],a[9]);
            s(a[3],a[4]);
            s(a[6],a[13]);
            s(a[8],a[14]);
            s(a[10],a[15]);
            s(a[11],a[12]);

            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            s(a[6],a[8]);
            s(a[7],a[9]);
            s(a[10],a[11]);
            s(a[12],a[13]);
            s(a[14],a[15]);

            s(a[0],a[2]);
            s(a[1],a[3]);
            s(a[4],a[10]);
            s(a[5],a[11]);
            s(a[6],a[7]);
            s(a[8],a[9]);
            s(a[12],a[14]);
            s(a[13],a[15]);

            s(a[1],a[2]);
            s(a[3],a[12]);
            s(a[4],a[6]);
            s(a[5],a[7]);
            s(a[8],a[10]);
            s(a[9],a[11]);
            s(a[13],a[14]);

            s(a[1],a[4]);
            s(a[2],a[6]);
            s(a[5],a[8]);
            s(a[7],a[10]);
            s(a[9],a[13]);
            s(a[11],a[14]);

            s(a[2],a[4]);
            s(a[3],a[6]);
            s(a[9],a[12]);
            s(a[11],a[13]);

            s(a[3],a[5]);
            s(a[6],a[8]);
            s(a[7],a[9]);
            s(a[10],a[12]);

            s(a[3],a[4]);
            s(a[5],a[6]);
            s(a[7],a[8]);
            s(a[9],a[10]);
            s(a[11],a[12]);

            s(a[6],a[7]);
            s(a[8],a[9]);
            
            
            
//            s(a[ 0],a[15]);
//            s(a[ 1],a[14]);
//            s(a[ 2],a[13]);
//            s(a[ 3],a[12]);
//            s(a[ 4],a[11]);
//            s(a[ 5],a[10]);
//            s(a[ 6],a[ 9]);
//            s(a[ 7],a[ 8]);
//            
//            s(a[ 0],a[ 7]);
//            s(a[ 1],a[ 6]);
//            s(a[ 2],a[ 5]);
//            s(a[ 3],a[ 4]);
//            
//            s(a[ 8],a[15]);
//            s(a[ 9],a[14]);
//            s(a[10],a[13]);
//            s(a[11],a[12]);
//            
//            s(a[ 0],a[ 3]);
//            s(a[ 1],a[ 2]);
//            s(a[ 4],a[ 7]);
//            s(a[ 5],a[ 6]);
//            s(a[ 8],a[11]);
//            s(a[ 9],a[10]);
//            s(a[12],a[15]);
//            s(a[13],a[14]);
//            
//            s(a[ 0],a[ 1]);
//            s(a[ 2],a[ 8]);
//            s(a[10],a[11]);
//            s(a[14],a[15]);
//            s(a[ 3],a[ 9]);
//            s(a[ 4],a[ 5]);
//            s(a[ 6],a[12]);
//            s(a[ 7],a[13]);
//            
//            
//            s(a[ 1],a[ 4]);
//            s(a[ 2],a[ 3]);
//            s(a[ 5],a[10]);
//            s(a[ 6],a[ 9]);
//            s(a[ 7],a[ 8]);
//            s(a[11],a[14]);
//            s(a[12],a[13]);
//            
//            s(a[ 1],a[ 2]);
//            s(a[ 3],a[ 4]);
//            s(a[ 5],a[ 8]);
//            s(a[11],a[12]);
//            s(a[13],a[14]);
//            s(a[ 6],a[10]);
//            s(a[ 7],a[ 9]);
//            
//            s(a[ 2],a[ 3]);
//            s(a[ 4],a[ 6]);
//            s(a[ 9],a[11]);
//            s(a[12],a[13]);
//            s(a[ 5],a[ 7]);
//            s(a[ 8],a[10]);
//            
//            s(a[ 4],a[ 5]);
//            s(a[ 6],a[ 7]);
//            s(a[ 8],a[ 9]);
//            s(a[10],a[11]);
//            
//            s(a[ 3],a[ 4]);
//            s(a[ 5],a[ 6]);
//            s(a[ 7],a[ 8]);
//            s(a[ 9],a[10]);
//            s(a[11],a[12]);
        }
    };
    
    // https://bertdobbelaere.github.io/sorting_networks.html#N32L185D14
    template<bool reverseQ>
    struct SortNet<32,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<32>( a, b );
            this->operator()( b, comp );
        }
        
        // This should be a depth-optimal solution.
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            s(a[6],a[7]);
            s(a[8],a[9]);
            s(a[10],a[11]);
            s(a[12],a[13]);
            s(a[14],a[15]);
            s(a[16],a[17]);
            s(a[18],a[19]);
            s(a[20],a[21]);
            s(a[22],a[23]);
            s(a[24],a[25]);
            s(a[26],a[27]);
            s(a[28],a[29]);
            s(a[30],a[31]);

            s(a[0],a[2]);
            s(a[1],a[3]);
            s(a[4],a[6]);
            s(a[5],a[7]);
            s(a[8],a[10]);
            s(a[9],a[11]);
            s(a[12],a[14]);
            s(a[13],a[15]);
            s(a[16],a[18]);
            s(a[17],a[19]);
            s(a[20],a[22]);
            s(a[21],a[23]);
            s(a[24],a[26]);
            s(a[25],a[27]);
            s(a[28],a[30]);
            s(a[29],a[31]);

            s(a[0],a[4]);
            s(a[1],a[5]);
            s(a[2],a[6]);
            s(a[3],a[7]);
            s(a[8],a[12]);
            s(a[9],a[13]);
            s(a[10],a[14]);
            s(a[11],a[15]);
            s(a[16],a[20]);
            s(a[17],a[21]);
            s(a[18],a[22]);
            s(a[19],a[23]);
            s(a[24],a[28]);
            s(a[25],a[29]);
            s(a[26],a[30]);
            s(a[27],a[31]);

            s(a[0],a[8]);
            s(a[1],a[9]);
            s(a[2],a[10]);
            s(a[3],a[11]);
            s(a[4],a[12]);
            s(a[5],a[13]);
            s(a[6],a[14]);
            s(a[7],a[15]);
            s(a[16],a[24]);
            s(a[17],a[25]);
            s(a[18],a[26]);
            s(a[19],a[27]);
            s(a[20],a[28]);
            s(a[21],a[29]);
            s(a[22],a[30]);
            s(a[23],a[31]);

            s(a[0],a[16]);
            s(a[1],a[8]);
            s(a[2],a[4]);
            s(a[3],a[12]);
            s(a[5],a[10]);
            s(a[6],a[9]);
            s(a[7],a[14]);
            s(a[11],a[13]);
            s(a[15],a[31]);
            s(a[17],a[24]);
            s(a[18],a[20]);
            s(a[19],a[28]);
            s(a[21],a[26]);
            s(a[22],a[25]);
            s(a[23],a[30]);
            s(a[27],a[29]);

            s(a[1],a[2]);
            s(a[3],a[5]);
            s(a[4],a[8]);
            s(a[6],a[22]);
            s(a[7],a[11]);
            s(a[9],a[25]);
            s(a[10],a[12]);
            s(a[13],a[14]);
            s(a[17],a[18]);
            s(a[19],a[21]);
            s(a[20],a[24]);
            s(a[23],a[27]);
            s(a[26],a[28]);
            s(a[29],a[30]);

            s(a[1],a[17]);
            s(a[2],a[18]);
            s(a[3],a[19]);
            s(a[4],a[20]);
            s(a[5],a[10]);
            s(a[7],a[23]);
            s(a[8],a[24]);
            s(a[11],a[27]);
            s(a[12],a[28]);
            s(a[13],a[29]);
            s(a[14],a[30]);
            s(a[21],a[26]);

            s(a[3],a[17]);
            s(a[4],a[16]);
            s(a[5],a[21]);
            s(a[6],a[18]);
            s(a[7],a[9]);
            s(a[8],a[20]);
            s(a[10],a[26]);
            s(a[11],a[23]);
            s(a[13],a[25]);
            s(a[14],a[28]);
            s(a[15],a[27]);
            s(a[22],a[24]);

            s(a[1],a[4]);
            s(a[3],a[8]);
            s(a[5],a[16]);
            s(a[7],a[17]);
            s(a[9],a[21]);
            s(a[10],a[22]);
            s(a[11],a[19]);
            s(a[12],a[20]);
            s(a[14],a[24]);
            s(a[15],a[26]);
            s(a[23],a[28]);
            s(a[27],a[30]);

            s(a[2],a[5]);
            s(a[7],a[8]);
            s(a[9],a[18]);
            s(a[11],a[17]);
            s(a[12],a[16]);
            s(a[13],a[22]);
            s(a[14],a[20]);
            s(a[15],a[19]);
            s(a[23],a[24]);
            s(a[26],a[29]);

            s(a[2],a[4]);
            s(a[6],a[12]);
            s(a[9],a[16]);
            s(a[10],a[11]);
            s(a[13],a[17]);
            s(a[14],a[18]);
            s(a[15],a[22]);
            s(a[19],a[25]);
            s(a[20],a[21]);
            s(a[27],a[29]);

            s(a[5],a[6]);
            s(a[8],a[12]);
            s(a[9],a[10]);
            s(a[11],a[13]);
            s(a[14],a[16]);
            s(a[15],a[17]);
            s(a[18],a[20]);
            s(a[19],a[23]);
            s(a[21],a[22]);
            s(a[25],a[26]);

            s(a[3],a[5]);
            s(a[6],a[7]);
            s(a[8],a[9]);
            s(a[10],a[12]);
            s(a[11],a[14]);
            s(a[13],a[16]);
            s(a[15],a[18]);
            s(a[17],a[20]);
            s(a[19],a[21]);
            s(a[22],a[23]);
            s(a[24],a[25]);
            s(a[26],a[28]);

            s(a[3],a[4]);
            s(a[5],a[6]);
            s(a[7],a[8]);
            s(a[9],a[10]);
            s(a[11],a[12]);
            s(a[13],a[14]);
            s(a[15],a[16]);
            s(a[17],a[18]);
            s(a[19],a[20]);
            s(a[21],a[22]);
            s(a[23],a[24]);
            s(a[25],a[26]);
            s(a[27],a[28]);
        }
    };
    
    // https://bertdobbelaere.github.io/sorting_networks_extended.html
    template<bool reverseQ>
    struct SortNet<64,reverseQ>
    {
        template<typename T, typename C = std::less<T>>
        void operator()( cptr<T> a, mptr<T> b, C && comp = C() )
        {
            copy_buffer<64>( a, b );
            this->operator()( b, comp );
        }
        
        // This should be a depth-optimal solution.
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, C && comp = C() )
        {
            auto s = [&comp]( T & x, T & y )
            {
                return CompSwap<reverseQ>(x,y,std::forward<C>(comp));
            };
            
            s(a[0],a[2]);
            s(a[1],a[3]);
            s(a[4],a[6]);
            s(a[5],a[7]);
            s(a[8],a[10]);
            s(a[9],a[11]);
            s(a[12],a[14]);
            s(a[13],a[15]);
            s(a[16],a[18]);
            s(a[17],a[19]);
            s(a[20],a[22]);
            s(a[21],a[23]);
            s(a[24],a[26]);
            s(a[25],a[27]);
            s(a[28],a[30]);
            s(a[29],a[31]);
            s(a[32],a[34]);
            s(a[33],a[35]);
            s(a[36],a[38]);
            s(a[37],a[39]);
            s(a[40],a[42]);
            s(a[41],a[43]);
            s(a[44],a[46]);
            s(a[45],a[47]);
            s(a[48],a[50]);
            s(a[49],a[51]);
            s(a[52],a[54]);
            s(a[53],a[55]);
            s(a[56],a[58]);
            s(a[57],a[59]);
            s(a[60],a[62]);
            s(a[61],a[63]);

            s(a[0],a[1]);
            s(a[2],a[3]);
            s(a[4],a[5]);
            s(a[6],a[7]);
            s(a[8],a[9]);
            s(a[10],a[11]);
            s(a[12],a[13]);
            s(a[14],a[15]);
            s(a[16],a[17]);
            s(a[18],a[19]);
            s(a[20],a[21]);
            s(a[22],a[23]);
            s(a[24],a[25]);
            s(a[26],a[27]);
            s(a[28],a[29]);
            s(a[30],a[31]);
            s(a[32],a[33]);
            s(a[34],a[35]);
            s(a[36],a[37]);
            s(a[38],a[39]);
            s(a[40],a[41]);
            s(a[42],a[43]);
            s(a[44],a[45]);
            s(a[46],a[47]);
            s(a[48],a[49]);
            s(a[50],a[51]);
            s(a[52],a[53]);
            s(a[54],a[55]);
            s(a[56],a[57]);
            s(a[58],a[59]);
            s(a[60],a[61]);
            s(a[62],a[63]);

            s(a[0],a[20]);
            s(a[1],a[2]);
            s(a[3],a[23]);
            s(a[4],a[16]);
            s(a[5],a[6]);
            s(a[7],a[19]);
            s(a[8],a[48]);
            s(a[9],a[10]);
            s(a[11],a[51]);
            s(a[12],a[52]);
            s(a[13],a[14]);
            s(a[15],a[55]);
            s(a[17],a[18]);
            s(a[21],a[22]);
            s(a[24],a[28]);
            s(a[25],a[26]);
            s(a[27],a[31]);
            s(a[29],a[30]);
            s(a[32],a[36]);
            s(a[33],a[34]);
            s(a[35],a[39]);
            s(a[37],a[38]);
            s(a[40],a[60]);
            s(a[41],a[42]);
            s(a[43],a[63]);
            s(a[44],a[56]);
            s(a[45],a[46]);
            s(a[47],a[59]);
            s(a[49],a[50]);
            s(a[53],a[54]);
            s(a[57],a[58]);
            s(a[61],a[62]);

            s(a[0],a[8]);
            s(a[1],a[21]);
            s(a[2],a[22]);
            s(a[3],a[11]);
            s(a[4],a[40]);
            s(a[5],a[17]);
            s(a[6],a[18]);
            s(a[7],a[43]);
            s(a[9],a[49]);
            s(a[10],a[50]);
            s(a[12],a[24]);
            s(a[13],a[53]);
            s(a[14],a[54]);
            s(a[15],a[27]);
            s(a[16],a[28]);
            s(a[19],a[31]);
            s(a[20],a[56]);
            s(a[23],a[59]);
            s(a[25],a[29]);
            s(a[26],a[30]);
            s(a[32],a[44]);
            s(a[33],a[37]);
            s(a[34],a[38]);
            s(a[35],a[47]);
            s(a[36],a[48]);
            s(a[39],a[51]);
            s(a[41],a[61]);
            s(a[42],a[62]);
            s(a[45],a[57]);
            s(a[46],a[58]);
            s(a[52],a[60]);
            s(a[55],a[63]);

            s(a[0],a[32]);
            s(a[1],a[9]);
            s(a[2],a[10]);
            s(a[3],a[35]);
            s(a[4],a[12]);
            s(a[5],a[41]);
            s(a[6],a[42]);
            s(a[7],a[15]);
            s(a[8],a[44]);
            s(a[11],a[47]);
            s(a[13],a[25]);
            s(a[14],a[26]);
            s(a[16],a[52]);
            s(a[17],a[29]);
            s(a[18],a[30]);
            s(a[19],a[55]);
            s(a[20],a[36]);
            s(a[21],a[57]);
            s(a[22],a[58]);
            s(a[23],a[39]);
            s(a[24],a[40]);
            s(a[27],a[43]);
            s(a[28],a[60]);
            s(a[31],a[63]);
            s(a[33],a[45]);
            s(a[34],a[46]);
            s(a[37],a[49]);
            s(a[38],a[50]);
            s(a[48],a[56]);
            s(a[51],a[59]);
            s(a[53],a[61]);
            s(a[54],a[62]);

            s(a[0],a[4]);
            s(a[1],a[33]);
            s(a[2],a[34]);
            s(a[3],a[7]);
            s(a[5],a[13]);
            s(a[6],a[14]);
            s(a[8],a[16]);
            s(a[9],a[45]);
            s(a[10],a[46]);
            s(a[11],a[19]);
            s(a[12],a[32]);
            s(a[15],a[35]);
            s(a[17],a[53]);
            s(a[18],a[54]);
            s(a[20],a[24]);
            s(a[21],a[37]);
            s(a[22],a[38]);
            s(a[23],a[27]);
            s(a[25],a[41]);
            s(a[26],a[42]);
            s(a[28],a[48]);
            s(a[29],a[61]);
            s(a[30],a[62]);
            s(a[31],a[51]);
            s(a[36],a[40]);
            s(a[39],a[43]);
            s(a[44],a[52]);
            s(a[47],a[55]);
            s(a[49],a[57]);
            s(a[50],a[58]);
            s(a[56],a[60]);
            s(a[59],a[63]);

            s(a[1],a[5]);
            s(a[2],a[6]);
            s(a[4],a[12]);
            s(a[7],a[15]);
            s(a[8],a[20]);
            s(a[9],a[17]);
            s(a[10],a[18]);
            s(a[11],a[23]);
            s(a[13],a[33]);
            s(a[14],a[34]);
            s(a[16],a[32]);
            s(a[19],a[35]);
            s(a[21],a[25]);
            s(a[22],a[26]);
            s(a[24],a[36]);
            s(a[27],a[39]);
            s(a[28],a[44]);
            s(a[29],a[49]);
            s(a[30],a[50]);
            s(a[31],a[47]);
            s(a[37],a[41]);
            s(a[38],a[42]);
            s(a[40],a[52]);
            s(a[43],a[55]);
            s(a[45],a[53]);
            s(a[46],a[54]);
            s(a[48],a[56]);
            s(a[51],a[59]);
            s(a[57],a[61]);
            s(a[58],a[62]);

            s(a[4],a[8]);
            s(a[5],a[13]);
            s(a[6],a[14]);
            s(a[7],a[11]);
            s(a[9],a[21]);
            s(a[10],a[22]);
            s(a[12],a[20]);
            s(a[15],a[23]);
            s(a[16],a[44]);
            s(a[17],a[33]);
            s(a[18],a[34]);
            s(a[19],a[47]);
            s(a[24],a[32]);
            s(a[25],a[37]);
            s(a[26],a[38]);
            s(a[27],a[35]);
            s(a[28],a[36]);
            s(a[29],a[45]);
            s(a[30],a[46]);
            s(a[31],a[39]);
            s(a[40],a[48]);
            s(a[41],a[53]);
            s(a[42],a[54]);
            s(a[43],a[51]);
            s(a[49],a[57]);
            s(a[50],a[58]);
            s(a[52],a[56]);
            s(a[55],a[59]);

            s(a[5],a[9]);
            s(a[6],a[10]);
            s(a[8],a[12]);
            s(a[11],a[15]);
            s(a[13],a[21]);
            s(a[14],a[22]);
            s(a[16],a[20]);
            s(a[17],a[45]);
            s(a[18],a[46]);
            s(a[19],a[23]);
            s(a[24],a[28]);
            s(a[25],a[33]);
            s(a[26],a[34]);
            s(a[27],a[31]);
            s(a[29],a[37]);
            s(a[30],a[38]);
            s(a[32],a[36]);
            s(a[35],a[39]);
            s(a[40],a[44]);
            s(a[41],a[49]);
            s(a[42],a[50]);
            s(a[43],a[47]);
            s(a[48],a[52]);
            s(a[51],a[55]);
            s(a[53],a[57]);
            s(a[54],a[58]);

            s(a[9],a[13]);
            s(a[10],a[14]);
            s(a[16],a[24]);
            s(a[17],a[21]);
            s(a[18],a[22]);
            s(a[19],a[27]);
            s(a[20],a[28]);
            s(a[23],a[31]);
            s(a[25],a[29]);
            s(a[26],a[30]);
            s(a[32],a[40]);
            s(a[33],a[37]);
            s(a[34],a[38]);
            s(a[35],a[43]);
            s(a[36],a[44]);
            s(a[39],a[47]);
            s(a[41],a[45]);
            s(a[42],a[46]);
            s(a[49],a[53]);
            s(a[50],a[54]);

            s(a[12],a[16]);
            s(a[15],a[19]);
            s(a[17],a[25]);
            s(a[18],a[26]);
            s(a[20],a[24]);
            s(a[21],a[29]);
            s(a[22],a[30]);
            s(a[23],a[27]);
            s(a[28],a[32]);
            s(a[31],a[35]);
            s(a[33],a[41]);
            s(a[34],a[42]);
            s(a[36],a[40]);
            s(a[37],a[45]);
            s(a[38],a[46]);
            s(a[39],a[43]);
            s(a[44],a[48]);
            s(a[47],a[51]);

            s(a[1],a[16]);
            s(a[2],a[32]);
            s(a[5],a[20]);
            s(a[6],a[36]);
            s(a[9],a[24]);
            s(a[10],a[40]);
            s(a[13],a[17]);
            s(a[14],a[18]);
            s(a[21],a[25]);
            s(a[22],a[26]);
            s(a[23],a[53]);
            s(a[27],a[57]);
            s(a[29],a[33]);
            s(a[30],a[34]);
            s(a[31],a[61]);
            s(a[37],a[41]);
            s(a[38],a[42]);
            s(a[39],a[54]);
            s(a[43],a[58]);
            s(a[45],a[49]);
            s(a[46],a[50]);
            s(a[47],a[62]);

            s(a[1],a[4]);
            s(a[2],a[8]);
            s(a[3],a[33]);
            s(a[6],a[12]);
            s(a[7],a[37]);
            s(a[10],a[24]);
            s(a[11],a[41]);
            s(a[13],a[28]);
            s(a[14],a[44]);
            s(a[15],a[45]);
            s(a[18],a[48]);
            s(a[19],a[49]);
            s(a[21],a[36]);
            s(a[22],a[52]);
            s(a[26],a[56]);
            s(a[27],a[42]);
            s(a[30],a[60]);
            s(a[35],a[50]);
            s(a[39],a[53]);
            s(a[51],a[57]);
            s(a[55],a[61]);
            s(a[59],a[62]);

            s(a[2],a[4]);
            s(a[3],a[17]);
            s(a[5],a[6]);
            s(a[7],a[22]);
            s(a[8],a[16]);
            s(a[11],a[25]);
            s(a[12],a[20]);
            s(a[14],a[28]);
            s(a[15],a[29]);
            s(a[18],a[32]);
            s(a[19],a[33]);
            s(a[23],a[37]);
            s(a[26],a[40]);
            s(a[30],a[44]);
            s(a[31],a[45]);
            s(a[34],a[48]);
            s(a[35],a[49]);
            s(a[38],a[52]);
            s(a[41],a[56]);
            s(a[43],a[51]);
            s(a[46],a[60]);
            s(a[47],a[55]);
            s(a[57],a[58]);
            s(a[59],a[61]);

            s(a[3],a[18]);
            s(a[7],a[21]);
            s(a[11],a[32]);
            s(a[15],a[30]);
            s(a[17],a[26]);
            s(a[19],a[25]);
            s(a[22],a[36]);
            s(a[23],a[29]);
            s(a[27],a[41]);
            s(a[31],a[52]);
            s(a[33],a[48]);
            s(a[34],a[40]);
            s(a[37],a[46]);
            s(a[38],a[44]);
            s(a[42],a[56]);
            s(a[45],a[60]);

            s(a[3],a[16]);
            s(a[7],a[20]);
            s(a[11],a[24]);
            s(a[15],a[21]);
            s(a[17],a[18]);
            s(a[19],a[34]);
            s(a[22],a[28]);
            s(a[23],a[38]);
            s(a[25],a[40]);
            s(a[26],a[32]);
            s(a[27],a[33]);
            s(a[29],a[44]);
            s(a[30],a[36]);
            s(a[31],a[37]);
            s(a[35],a[41]);
            s(a[39],a[52]);
            s(a[42],a[48]);
            s(a[43],a[56]);
            s(a[45],a[46]);
            s(a[47],a[60]);

            s(a[3],a[9]);
            s(a[7],a[13]);
            s(a[10],a[16]);
            s(a[11],a[17]);
            s(a[14],a[20]);
            s(a[15],a[22]);
            s(a[18],a[24]);
            s(a[19],a[26]);
            s(a[21],a[28]);
            s(a[23],a[30]);
            s(a[25],a[32]);
            s(a[27],a[34]);
            s(a[29],a[36]);
            s(a[31],a[38]);
            s(a[33],a[40]);
            s(a[35],a[42]);
            s(a[37],a[44]);
            s(a[39],a[45]);
            s(a[41],a[48]);
            s(a[43],a[49]);
            s(a[46],a[52]);
            s(a[47],a[53]);
            s(a[50],a[56]);
            s(a[54],a[60]);

            s(a[3],a[8]);
            s(a[7],a[10]);
            s(a[9],a[12]);
            s(a[11],a[16]);
            s(a[13],a[14]);
            s(a[15],a[17]);
            s(a[18],a[20]);
            s(a[19],a[22]);
            s(a[21],a[24]);
            s(a[23],a[26]);
            s(a[25],a[28]);
            s(a[27],a[29]);
            s(a[30],a[32]);
            s(a[31],a[33]);
            s(a[34],a[36]);
            s(a[35],a[38]);
            s(a[37],a[40]);
            s(a[39],a[42]);
            s(a[41],a[44]);
            s(a[43],a[45]);
            s(a[46],a[48]);
            s(a[47],a[52]);
            s(a[49],a[50]);
            s(a[51],a[54]);
            s(a[53],a[56]);
            s(a[55],a[60]);

            s(a[3],a[5]);
            s(a[6],a[8]);
            s(a[7],a[9]);
            s(a[10],a[12]);
            s(a[11],a[13]);
            s(a[14],a[16]);
            s(a[15],a[18]);
            s(a[17],a[20]);
            s(a[19],a[21]);
            s(a[22],a[24]);
            s(a[23],a[25]);
            s(a[26],a[28]);
            s(a[27],a[30]);
            s(a[29],a[32]);
            s(a[31],a[34]);
            s(a[33],a[36]);
            s(a[35],a[37]);
            s(a[38],a[40]);
            s(a[39],a[41]);
            s(a[42],a[44]);
            s(a[43],a[46]);
            s(a[45],a[48]);
            s(a[47],a[49]);
            s(a[50],a[52]);
            s(a[51],a[53]);
            s(a[54],a[56]);
            s(a[55],a[57]);
            s(a[58],a[60]);

            s(a[3],a[4]);
            s(a[5],a[6]);
            s(a[7],a[8]);
            s(a[9],a[10]);
            s(a[11],a[12]);
            s(a[13],a[14]);
            s(a[15],a[16]);
            s(a[17],a[18]);
            s(a[19],a[20]);
            s(a[21],a[22]);
            s(a[23],a[24]);
            s(a[25],a[26]);
            s(a[27],a[28]);
            s(a[29],a[30]);
            s(a[31],a[32]);
            s(a[33],a[34]);
            s(a[35],a[36]);
            s(a[37],a[38]);
            s(a[39],a[40]);
            s(a[41],a[42]);
            s(a[43],a[44]);
            s(a[45],a[46]);
            s(a[47],a[48]);
            s(a[49],a[50]);
            s(a[51],a[52]);
            s(a[53],a[54]);
            s(a[55],a[56]);
            s(a[57],a[58]);
            s(a[59],a[60]);
        }
    };

} // namespace Tools
