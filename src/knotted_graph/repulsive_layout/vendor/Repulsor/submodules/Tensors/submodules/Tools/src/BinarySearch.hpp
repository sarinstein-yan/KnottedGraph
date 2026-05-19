#pragma once

namespace Tools
{
    /*!@brief Returns the first position `p` in the sorted input array `sorted_list` such that `sorted_list[p] > value`. If no such position exists, then `p = n`is returned.
     */
    template<typename T, IntQ Int, typename CMP = std::less<T>>
    inline Int FindFirstPositionGreater(
        cptr<T> sorted_list, const Int n, cref<T> value, CMP && cmp = CMP()
    )
    {
        if( n <= Int(0) ) { return Int(0); }
        
        Int L = Int(0);
        
        if( cmp(value,sorted_list[L]) ) // sorted_list[L] > value
        {
            return L;
        }
        
        Int R = n - Int(1);
        
        if( !cmp(value,sorted_list[R]) ) // sorted_list[R] <= value
        {
            return n;
        }

        // Invariants from here on: sorted_list[L] <= value < sorted_list[R]
        // Starting binary search.
        while( L + Int(1) < R )
        {
            const Int C = R - (R-L)/static_cast<Int>(2);

            if( cmp(value,sorted_list[C]) ) // value < sorted_list[C]
            {
                R = C;
            }
            else // sorted_list[C] <= value
            {
                L = C;
            }
        }
        
        return R;
        
    } // FindFirstPositionGreater
    
    
    /*!@brief Returns the first position `p` in the sorted input array `sorted_list` such that `sorted_list[p] >= value`. If no such position exists, then `p = n`is returned.
     */
    template<typename T, IntQ Int, typename CMP = std::less<T>>
    inline Int FindFirstPositionGreaterEqual(
        cptr<T> sorted_list, const Int n, cref<T> value, CMP && cmp = CMP()
    )
    {
        if( n <= Int(0) ) { return Int(0); }
        
        Int L = Int(0);
        
        if( !cmp(sorted_list[L],value) ) // sorted_list[L] >= value
        {
            return L;
        }
        
        Int R = n - Int(1);
        
        if( cmp(sorted_list[R],value) ) // sorted_list[R] < value
        {
            return n;
        }

        // Invariants from here on: sorted_list[L] < value <= sorted_list[R]
        // Starting binary search.
        while( L + Int(1) < R )
        {
            const Int C = R - (R-L)/static_cast<Int>(2);

            if( !cmp(sorted_list[C],value) ) // value <= sorted_list[C]
            {
                R = C;
            }
            else // sorted_list[C] <= value
            {
                L = C;
            }
        }
        
        return R;
        
    } // FindFirstPositionGreaterEqual
    
    
    /*!@brief Returns the largest integer interval `[begin,end[` in `[0,n]`  such that each `i`, `begin <= i < end` satisfies `a <= sorted_list[i] < b`. The empty interval is represented by`[n,n[`.
     * This function requires that `sorted_list` is a sorted array of length at least `n` without duplicates.
     */
    
    template<typename T, IntQ Int, typename CMP = std::less<T>>
    inline std::pair<Int,Int> FindRange(
        cptr<T> sorted_list, const Int n, cref<T> a, cref<T> b, CMP && cmp = CMP()
    )
    {
//        print("FindRange");
//        TOOLS_DUMP(a);
//        TOOLS_DUMP(b);
//        TOOLS_DUMP(n);
        
        if( !cmp(a,b) ) { return std::pair{Int(0),Int(0)}; }
        
        Int end = FindFirstPositionGreaterEqual(sorted_list,n,b,cmp);
        
//        Int begin = 0;
//        if ( cmp(sorted_list[begin],a) ) // a < sorted_list[begin]
//        {
//            // The correct value for begin must be greater than 0.
//            begin = FindFirstPositionGreater(sorted_list,end,a,cmp);
//            if( !cmp(sorted_list[begin-Int(1)],a) )
//            {
//                --begin;
//            }
//        }
        
        Int begin = FindFirstPositionGreaterEqual(sorted_list,end,a,cmp);

//        TOOLS_DUMP(begin);
//        TOOLS_DUMP(a);
//        TOOLS_DUMP(sorted_list[begin]);
//        TOOLS_DUMP(sorted_list[begin+1]);
//        TOOLS_DUMP(end);
//        TOOLS_DUMP(b);
//        TOOLS_DUMP(sorted_list[end-1]);
//        TOOLS_DUMP(sorted_list[end]);
        
        return std::pair{begin,end};
        
    } // FindRange
    
    
//    /*!@brief Returns the first position `pos` in the sorted input array `sorted_list` such that `sorted_list[pos] >= value`. If such a position exists, then `std::pair(true,pos)`is returned. Otherwise `std::pair(false,n)` is returned.
//     */
//    template<typename T, IntQ Int, typename CMP = std::less<T>>
//    inline std::pair<bool,Int> FindFirstPositionGreaterEqual(
//        cptr<T> sorted_list, const Int n, cref<T> value, CMP && cmp = CMP()
//    )
//    {
//        Int p = FindFirstPositionGreater(sorted_list,n,value,cmp);
//        
//        if( (p > Int(0)) && !cmp(value,sorted_list[p-Int(1)]) )
//        {
//            return p - Int(1);
//        }
//        else
//        {
//            return p;
//        }
//        
//    } // FindFirstPositionGreaterEqual
    

    //    /*!@brief Returns the largest position `pos` in the input array such that `sorted_list[pos] <= value`. If `value` is found, then `true` is returned. Otherwise `false` is returned.
    //     */
    //    template<typename T, IntQ Int>
    //    TOOLS_FORCE_INLINE bool BinarySearch(
    //        cptr<T> sorted_list, const Int n, cref<T> value, mref<Int> pos
    //    )
    //    {
    //        if( n == Int(0) )
    //        {
    //            pos = Int(0);
    //            return false;
    //        }
    //
    //        Int L = Int(0);
    //        Int R = n - Int(1);
    //
    //        if( value <= sorted_list[L] )
    //        {
    //            pos = L;
    //            return (value == sorted_list[L]);
    //        }
    //
    //        if( value >= sorted_list[R] )
    //        {
    //            pos = R;
    //            return (value == sorted_list[R]);
    //        }
    //
    //        while( L < R )
    //        {
    //            const Int C = R - (R-L)/static_cast<Int>(2);
    //
    //            if( value < sorted_list[C] )
    //            {
    //                R = C - 1;
    //            }
    //            else
    //            {
    //                L = C;
    //            }
    //        }
    //
    //        pos = L;
    //        return (sorted_list[L] == value);
    //
    //    } // BinarySearch
    
} // namespace Tools
