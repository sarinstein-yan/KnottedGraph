#pragma once

namespace Tensors
{
    template<typename S, typename T, IntQ I>
    class TwoArrayQuickSort
    {
    protected:
        
        Tensor1<I,I> stack;
        I stackptr = 0;
        
    public:
        
        void operator()( mptr<S> a, mptr<T> b, const I n, const bool reverse = false )
        {
            if( n <= I(1) )
            {
                return;
            }            

            // a - the list that is to be sorted.
            // b - the list to which each move of a shall also be applied.
            
            // Recursion-free implementation from https://www.geeksforgeeks.org/iterative-quick-sort/
            // The handling of duplicates is taken from https://cs.stackexchange.com/a/104825
            if( I(2) * n + I(2) > stack.Size() )
            {
                stack = Tensor1<I,I>( I(2) * n + I(2) );
            }
            
            stackptr = 0;
            
            stack[++stackptr] = 0;
            stack[++stackptr] = n-1;
            
            if( reverse )
            {
                while( stackptr > I(1) )
                {
                    const I hi = stack[stackptr--];
                    const I lo = stack[stackptr--];
                    
                    I l = lo;
                    I r = lo;
                    I u = hi;
                    
                    // lo <= l <= r <= u <= hi
                    
                    // - elements in [lo,l) are smaller than pivot
                    // - elements in [l,r) are equal to pivot
                    // - elements in [r,u] are not examined, yet
                    // - elements in (u,hi] are greater than pivot
                    
                    const S pivot = a[lo];
                    
                    while( r <= u )
                    {
                        if( a[r] > pivot )
                        {
                            std::swap( a[l], a[r] );
                            std::swap( b[l], b[r] );
                            l++;
                            r++;
                        }
                        else if( a[r] < pivot )
                        {
                            std::swap( a[r], a[u] );
                            std::swap( b[r], b[u] );
                            u--;
                        }
                        else
                        {
                            // element a[r] is equal to pivot
                            r++;
                        }
                    }
                    
                    if( l > lo + I(1) )
                    {
                        stack[++stackptr] = lo;
                        stack[++stackptr] = l-1;
                    }
                    
                    if( r < hi )
                    {
                        stack[++stackptr] = r;
                        stack[++stackptr] = hi;
                    }
                }
            }
            else
            {
                while( stackptr > I(1) )
                {
                    const I hi = stack[stackptr--];
                    const I lo = stack[stackptr--];
                    
                    I l = lo;
                    I r = lo;
                    I u = hi;
                    
                    // lo <= l <= r <= u <= hi
                    
                    // - elements in [lo,l) are smaller than pivot
                    // - elements in [l,r) are equal to pivot
                    // - elements in [r,u] are not examined, yet
                    // - elements in (u,hi] are greater than pivot
                    
                    const S pivot = a[lo];
                    
                    while( r <= u )
                    {
                        if( a[r] < pivot )
                        {
                            std::swap( a[l], a[r] );
                            std::swap( b[l], b[r] );
                            l++;
                            r++;
                        }
                        else if( a[r] > pivot )
                        {
                            std::swap( a[r], a[u] );
                            std::swap( b[r], b[u] );
                            u--;
                        }
                        else
                        {
                            // element a[r] is equal to pivot
                            r++;
                        }
                    }
                    
                    
                    if( l > lo + 1 )
                    {
                        stack[++stackptr] = lo;
                        stack[++stackptr] = l-1;
                    }
                    
                    if( r < hi )
                    {
                        stack[++stackptr] = r;
                        stack[++stackptr] = hi;
                    }
                }
            }
        }
    }; // TwoArrayQuickSort
    
}
