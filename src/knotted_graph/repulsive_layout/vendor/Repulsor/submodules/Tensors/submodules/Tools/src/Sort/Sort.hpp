#pragma once

#include <functional>

namespace Tools
{
    template<bool reverseQ = false, Size_T sortnet_limit = 16, typename T, typename C = std::less<T>>
    void Sort( mptr<T> begin, mptr<T> end, C && comp = C() )
    {
        std::ptrdiff_t n = std::distance( begin, end );

        if( n <= std::ptrdiff_t(1) ) { return; }
    
        // Use sorting networks for inputs of length <= 16; use std::sort otherwise.
        switch( n )
        {
            case 2:
            {
                if constexpr ( sortnet_limit >= 2 )
                {
                    return SortNet<2,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 3:
            {
                if constexpr ( sortnet_limit >= 3 )
                {
                    return SortNet<3,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 4:
            {
                if constexpr ( sortnet_limit >= 4 )
                {
                    return SortNet<4,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 5:
            {
                if constexpr ( sortnet_limit >= 5 )
                {
                    return SortNet<5,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 6:
            {
                if constexpr ( sortnet_limit >= 6 )
                {
                    return SortNet<6,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 7:
            {
                if constexpr ( sortnet_limit >= 7 )
                {
                    return SortNet<7,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 8:
            {
                if constexpr ( sortnet_limit >= 8 )
                {
                    return SortNet<8,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 9:
            {
                if constexpr ( sortnet_limit >= 9 )
                {
                    return SortNet<9,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 10:
            {
                if constexpr ( sortnet_limit >= 10 )
                {
                    return SortNet<10,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 11:
            {
                if constexpr ( sortnet_limit >= 11 )
                {
                    return SortNet<11,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 12:
            {
                if constexpr ( sortnet_limit >= 12 )
                {
                    return SortNet<12,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 13:
            {
                if constexpr ( sortnet_limit >= 13 )
                {
                    return SortNet<13,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 14:
            {
                if constexpr ( sortnet_limit >= 14 )
                {
                    return SortNet<14,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 15:
            {
                if constexpr ( sortnet_limit >= 15 )
                {
                    return SortNet<15,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 16:
            {
                if constexpr ( sortnet_limit >= 16 )
                {
                    return SortNet<16,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 32:
            {
                if constexpr ( sortnet_limit >= 32 )
                {
                    return SortNet<32,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            case 64:
            {
                if constexpr ( sortnet_limit >= 64 )
                {
                    return SortNet<64,reverseQ>()(begin,std::forward<C>(comp));
                }
                else
                {
                    return std::sort( begin, end, std::forward<C>(comp) );
                }
            }
            default:
            {
                return std::sort( begin, end, std::forward<C>(comp) );
            }
        }
    }

} // namespace Tools

