#pragma once

namespace Tensors
{
    template<typename T, typename I, I RUN = 32 >
    class TimSort final
    {
    public:
        
        TimSort() = default;
        
        explicit TimSort( const I estimated_size )
        :   buffer( estimated_size )
        {}
        
    private:
        
        Tensor1<T,I> buffer { RUN };
        
        // This function sorts array from left index to
        // to right index which is of size atmost RUN
        void InsertionSort( mptr<T> a, const I left, const I right )
        {
            for( I i = left + 1; i <= right; ++i )
            {
                T temp = a[i];
                I j = i - 1;
                while( j >= left && a[j] > temp )
                {
                    a[j+1] = a[j];
                    j--;
                }
                a[j+1] = temp;
            }
        }
         
        // Merge function merges the sorted runs
        void Merge( mptr<T> a, const I l, const I m, const I r )
        {
            // Original array is broken in two parts L and R array.
            I L_size = m - l + 1;
            I R_size = r - m;
            mptr<I> L = &buffer.data()[0];
            mptr<I> R = &buffer.data()[L_size];
            
            copy_buffer( &a[l],   L, L_size );
            copy_buffer( &a[m+1], R, R_size );
         
            I i = 0;
            I j = 0;
            I k = l;
         
            // After comparing, we merge those two array in larger sub array.
            while( i < L_size && j < R_size )
            {
                if( L[i] <= R[j] )
                {
                    a[k] = L[i];
                    ++i;
                }
                else
                {
                    a[k] = R[j];
                    ++j;
                }
                ++k;
            }
         
            // Copy remaining elements of left, if any
            while( i < L_size )
            {
                a[k] = L[i];
                ++k;
                ++i;
            }
         
            // Copy remaining element of right, if any
            while( j < R_size )
            {
                a[k] = R[j];
                k++;
                j++;
            }
        }
         
    public:
        
        void operator()( mptr<T> begin, cptr<T> end )
        {
            const I n = end >= begin ? end - begin : I(0);
            
            operator()( begin, static_cast<I>(2)*n );
        }
        
        void operator()( mptr<T> a, const I n )
        {
            // https://www.geeksforgeeks.org/timsort/
            
            // Iterative Timsort function to sort the
            // array[0...n-1] (similar to merge sort)
            
            if( n > buffer.Size() )
            {
                buffer = Tensor1<T,I>(n);
            }
            
            // Sort individual subarrays of size RUN
            for( I i = 0; i < n; i+=RUN )
            {
                InsertionSort(a, i, Min( (i+RUN-1), (n-1)) );
            }
         
            // Start merging from size RUN (or 32).
            // It will merge to form size 2*RUN, then 4*RUN, 8*RUN and so on ....
            for( I size = RUN; size < n; size = static_cast<I>(2)*size )
            {
                
                I chunk_size = static_cast<I>(2)*size;
                
                // Pick starting point of left sub array.
                // We are going to merge arr[left..left+size-1] and arr[left+size, left+2*size-1].
                // After every merge, we increase left by 2*size.
                for( I l = 0; l < n; l += chunk_size )
                {
                    // find ending point of
                    // left sub array
                    // mid+1 is starting point
                    // of right sub array
                    I m = l + size - 1;
                    I r = Min( (l + chunk_size - 1), (n-1) );
         
                    // merge sub array arr[left.....mid] &
                    // arr[mid+1....right]
                      if( m < r )
                      {
                        Merge(a, l, m, r);
                      }
                }
            }
        }
        
    }; // TimSort

} // namspace Tensors
