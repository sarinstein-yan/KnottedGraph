#pragma once

namespace Tensors
{
    template<typename T, IntQ Int>
    bool NextPermutation( mptr<T> A, mptr<Int> p, const Int m, const Int n )
    {
        // This functions is supposed to do the same as std::next_permutation, but it is designed to carry along the swaps through the m x n matrix A.
        
        
        // The array p is supposed to be a permutation of integers {0,...,m-1}.
        // The array A is supposed to represent a matrix of size m x n.
        // We suppose that A is a row permutation of a matrix A0, i.e.
        // A = A0[ {p[0],p[1],...,p[m-1]}, : ].
        
        // Return value is true if output permutation is lexicographically greater than the input permutation.
        // That is, the return value is false if and only if p is ordered on return.
        
        // See https://www.geeksforgeeks.org/next-permutation/#
            
        if( m < Int(2) )
        {
            return true;
        }
        
        // Swaps i-th and j-th row of a.
        auto Swap = [=]( const Int i, const Int j )
        {
            std::swap( p[i], p[j] );
            
            std::swap_ranges( &A[n*i], &A[n*(i+Int(1))], &A[n*j] );
        };
        
        // Find pivot, i.e., the first position such that p[pivot+1],...,p[m] is descending.
        Int pivot = m - Int(2);
        
        Int a = p[pivot         ];
        Int b = p[pivot + Int(1)];
        
        while( (pivot > Int(0)) && (a > b) )
        {
            --pivot;
            b = a;
            a = p[pivot];
        }
        
        const bool pivot_foundQ = ( a < b );
        
        if( pivot_foundQ )
        {
            // Find rightmost successor of a = p[pivot] right to pivot. (There must be one.)
            // We can apply binary search here as p[pivot+1],...,p[m] is descending!

            Int successor;
            
            Int R = m - Int(1);
            Int p_R = p[R];

            
            if( a < p_R )
            {
                successor = R;
            }
            else
            {
                Int L   = pivot + Int(1);
                Int p_L = p[L];
                
                // We know that p[R] < p[pivot] < p[L].
                
                while( L + Int(1) < R )
                {
                    Int C   = L + (R-L) / Int(2);
                    Int p_C = p[C];

                    if( p_C  > a )
                    {
                        L   = C;
                        p_L = p_C;
                    }
                    else
                    {
                        R   = C;
                        p_R = p_C;
                    }
                }

                successor = L;
            }
            
            
            // Swap pivot with successor.
            Swap( pivot, successor );
        }
        
        // Reverse everything after pivot.
        {
            Int L = pivot + pivot_foundQ;
            Int R = m - Int(1);
            
            while( L < R )
            {
                Swap(L,R);
                ++L;
                --R;
            }
        }
        
        return pivot_foundQ;
    }
    
} // namespace Tensors
