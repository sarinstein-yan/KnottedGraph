#pragma once

namespace Tools {
    
    // https://stackoverflow.com/a/9729747/8248900

    template<IntQ T>
    struct hash
    {
        using is_avalanching = std::true_type; // instruct Boost.Unordered to not use post-mixing
        
        inline std::size_t operator()( cref<T> x_0 ) const
        {
            std::size_t x = static_cast<std::size_t>(x_0);
            x = (x ^ (x >> 30)) * std::size_t(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)) * std::size_t(0x94d049bb133111eb);
            x =  x ^ (x >> 31);
            return static_cast<std::size_t>(x);
        }
    };
    
    template <class T>
    inline void hash_combine(std::size_t & seed, const T & x)
    {
        using namespace std;
        hash<T> hasher;
        seed ^= hasher(x) + std::size_t(0x9e3779b9) + (seed << 6) + (seed >> 2);
    }

    template<typename S, typename T>
    struct pair_hash
    {
        using is_avalanching [[maybe_unused]] = std::true_type; // instruct Boost.Unordered to not use post-mixing
        
        inline std::size_t operator()(const std::pair<S,T> & p) const
        {
            std::size_t seed = 0;
            hash_combine(seed, p.first);
            hash_combine(seed, p.second);
            return seed;
        }
    };
    
    
    struct array_hash
    {
        using is_avalanching [[maybe_unused]] = std::true_type; // instruct Boost.Unordered to not use post-mixing
        
        template<typename T, std::size_t n>
        inline std::size_t operator()(const std::array<T,n> & a) const
        {
            std::size_t seed = 0;
            for( std::size_t i = 0; i < n; ++i )
            {
                hash_combine(seed, a[i]);
            }
            return seed;
        }
    };

    
} // namespace Tools
