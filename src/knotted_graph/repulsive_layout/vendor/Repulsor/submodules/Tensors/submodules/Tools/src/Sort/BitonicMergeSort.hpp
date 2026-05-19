#pragma once

namespace Tools
{
    // Merge the sorted arrays [a[0],...,a[m][ and [b[0],...,b[n][
    // into the array [c[0],...,c[m+n][ to obtain sorted array.
    template<bool reverseQ, typename T, typename C = std::less<T>>
    void MergeBuffers(
        cptr<T> a, const Size_T m, cptr<T> b, const Size_T n, mptr<T> c, C comp = C()
    )
    {
        Size_T i = 0;
        Size_T j = 0;
        Size_T k = 0;
        
        while( i < m && j < n )
        {
            if( comp(a[i],b[j]) != reverseQ )
            {
                c[k++] = a[i++];
            }
            else
            {
                c[k++] = b[j++];
            }
        }
     
        // Copy remaining elements of left, if any
        while( i < m )
        {
            c[k++] = a[i++];
        }
     
        // Copy remaining element of right, if any
        while( j < n )
        {
            c[k++] = b[j++];
        }

    }
    
    template<bool reverseQ, typename T, typename C = std::less<T>>
    void MergeBuffers(
        cptr<T> a_begin, cptr<T> a_end,
        cptr<T> b_begin, cptr<T> b_end,
        mptr<T> c_begin, mptr<T> c_end,
        C comp = C()
    )
    {
        const T * a = a_begin;
        const T * b = b_begin;
        T * c = c_begin;
        
        while( a < a_end && b < b_end )
        {
            if( comp(*a,*b) != reverseQ )
            {
                *(c++) = *(a++);
            }
            else
            {
                *(c++) = *(b++);
            }
        }
     
        // Copy remaining elements of left, if any
        while( a < a_end )
        {
            *(c++) = *(a++);
        }
     
        // Copy remaining element of right, if any
        while( b < b_end )
        {
            *(c++) = *(b++);
        }
    }
    
    
    template<Size_T vector_size = 8, bool reverseQ = false>
    class BitonicMergeSorter
    {
    public:
        
        BitonicMergeSorter() = default;
        
        ~BitonicMergeSorter() = default;
        
    private:
        
    public:
        
        template<typename T, typename C = std::less<T>>
        void operator()( mptr<T> a, mptr<T> b, const Size_T n, C comp = C() )
        {
            // a is the buffer to be sorted.
            // b is work space
            
            if( n <= 1 )
            {
                return;
            }
            
            FindPowerOfTwo<2>( a, b, n, comp  );
        }
                
        template<Size_T power_of_2, typename T, typename C = std::less<T>>
        void BitonicMergeSort( mptr<T> a, mptr<T> b, const Size_T n, C comp = C() )
        {
            TOOLS_PTIMER(timer,ClassName()+"::BitonicMergeSort<"+ToString(power_of_2)+","+TypeName<T>+">");
            
            BitonicSort<power_of_2,vector_size,reverseQ>( a, comp );
            
            if( n > power_of_2 )
            {
                T * a_ = &a[power_of_2];
                T * b_ = &b[power_of_2];
                
                Size_T n_ = n - power_of_2;
                
                if constexpr( power_of_2 > Size_T(16) )
                {
                    BitonicMergeSorter<vector_size,reverseQ> S ;
                    
                    S.BitonicMergeSort<power_of_2/Size_T(2)>( a_, b_, n_, comp );
                }
                else
                {
                   switch( n_ )
                    {
                        case 2:
                        {
                            SortNet<2,reverseQ>()(a_,comp);
                            break;
                        }
                        case 3:
                        {
                            SortNet<3,reverseQ>()(a_,comp);
                            break;
                        }
                        case 4:
                        {
                            SortNet<4,reverseQ>()(a_,comp);
                            break;
                        }
                        case 5:
                        {
                            SortNet<5,reverseQ>()(a_,comp);
                            break;
                        }
                        case 6:
                        {
                            SortNet<6,reverseQ>()(a_,comp);
                            break;
                        }
                        case 7:
                        {
                            SortNet<7,reverseQ>()(a_,comp);
                            break;
                        }
                        case 8:
                        {
                            SortNet<8,reverseQ>()(a_,comp);
                            break;
                        }
                        case 9:
                        {
                            SortNet<9,reverseQ>()(a_,comp);
                            break;
                        }
                        case 10:
                        {
                            SortNet<10,reverseQ>()(a_,comp);
                            break;
                        }
                        case 11:
                        {
                            SortNet<11,reverseQ>()(a_,comp);
                            break;
                        }
                        case 12:
                        {
                            SortNet<12,reverseQ>()(a_,comp);
                            break;
                        }
                        case 13:
                        {
                            SortNet<13,reverseQ>()(a_,comp);
                            break;
                        }
                        case 14:
                        {
                            SortNet<14,reverseQ>()(a_,comp);
                            break;
                        }
                        case 15:
                        {
                            SortNet<15,reverseQ>()(a_,comp);
                            break;
                        }
                        case 16:
                        {
                            SortNet<16,reverseQ>()(a_,comp);
                            break;
                        }
                    }
                }
                
                // TODO: This copy can be avoided by sorting the chunks directly into a, b, depending on the parity of the the depth of their leave node.
                
                copy_buffer( a, b, n );

                MergeBuffers<reverseQ>(
                    &b[0],          &b[power_of_2],
                    &b[power_of_2], &b[n],
                    &a[0],          &a[n],
                    comp
                );
            }
        }

        
    private:
        
        
        template<Size_T power_of_2, typename T, typename C>
        void FindPowerOfTwo( mptr<T> a, mptr<T> b, const Size_T n_, C comp  )
        {
            if( n_ < Size_T(2) * power_of_2 )
            {
                if constexpr ( power_of_2 < 2305843009213693952UL )
                {
                    return BitonicMergeSort<power_of_2>( a, b, n_, comp );
                }
            }
            else
            {
                if constexpr ( power_of_2 < 2305843009213693952UL )
                {
                    return FindPowerOfTwo<2*power_of_2>( a, b, n_, comp );
                }
            }
        }
        
        
        static std::string ClassName()
        {
            return std::string("BitonicMergeSorter")+"<"+ToString(vector_size)+","+ToString(reverseQ)+">";
        }
    };
    
} // namespace Tools


