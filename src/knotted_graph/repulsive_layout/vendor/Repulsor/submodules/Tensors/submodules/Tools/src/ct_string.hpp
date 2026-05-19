namespace Tools
{
    
    template<std::size_t N>
    struct ct_string
    {
        // Taken from https://stackoverflow.com/a/77803192/8248900
        
        static_assert( N > std::size_t(0) );
        
    private:
        
        char bytes[N];
        
    public:
        
        constexpr ct_string()
        {
            for( std::size_t i = std::size_t(0); i < N; ++i )
            {
                bytes[i] = std::size_t(0);
            }
        }
        
        constexpr ct_string( const ct_string & rhs )
        {
            for( std::size_t i = std::size_t(0); i < N; ++i )
            {
                bytes[i] = rhs[i];
            }
        }
        
        // Construct from a "string literal":
        constexpr ct_string( const char(&arr)[N] )
        {
            for( std::size_t i = 0; i < N; ++i )
            {
                bytes[i] = arr[i];
            }
        }
        
        // does not include trailing nil
        // so ct_string<10> has a max size of 9
        // if there is an earlier nil character, size()
        // is the length up to that nil
        [[nodiscard]] constexpr std::size_t size() const
        {
            std::size_t r = 0;
            
            while( r + 1 < N && bytes[r] )
            {
                ++r;
            }
            return r;
        }
        
        [[nodiscard]] constexpr std::size_t byte_count() const
        {
            return N;
        }
        
        
//        constexpr char & operator[](std::size_t i)
//        {
//            return bytes[i];
//        }
//        
//        constexpr char const & operator[](std::size_t i) const
//        {
//            return bytes[i];
//        }

        template<IntQ Int>
        constexpr char & operator[](const Int i)
        {
            return bytes[i];
        }
        
        template<IntQ Int>
        constexpr char const & operator[](const Int i) const
        {
            return bytes[i];
        }
        
        constexpr char const * data() const
        {
            return bytes;
        }
        
        constexpr operator char const*() const
        {
            return data();
        }
        
    };
    
    
    template<std::size_t M, std::size_t N>
    [[nodiscard]] constexpr ct_string<M+N-1> operator+( ct_string<M> lhs, ct_string<N> rhs )
    {
        ct_string<M+N-1> result;
        // copy up to first nil in lhs:
        for (std::size_t i = 0; i < lhs.size(); ++i)
        {
            result[i] = lhs[i];
        }
        // copy entire rhs buffer, including trailing nils:
        for (std::size_t i = 0; i < N; ++i)
        {
            result[lhs.size()+i] = rhs[i];
        }
        
        // zero out the leftovers, if any:
        for( std::size_t i = lhs.size() + N; i < M+N-1; ++i )
        {
            result[i] = 0;
        }
        
        return result;
    }
    
    
    template<std::size_t M, std::size_t N>
    [[nodiscard]] constexpr ct_string<M+N-1> operator+( ct_string<M> lhs, const char(&rhs)[N] )
    {
        return lhs + ct_string<N>( rhs );
    }
    
    template<std::size_t N>
    [[nodiscard]] std::string operator+( ct_string<N> lhs, const std::string & rhs )
    {
        return lhs.data() + rhs;
    }
    
    template<std::size_t N>
    [[nodiscard]] std::string operator+( const std::string & lhs, ct_string<N> rhs  )
    {
        return lhs + rhs.data();
    }
    
    template<Size_T t >
    constexpr ct_string<t + 1> ct_tabs = ct_tabs<t-1> + ct_string<2>("\t");
    template<> constexpr ct_string<1> ct_tabs<0> = ct_string<1>("");
    
    template<Size_T t >
    constexpr ct_string<t + 1> ct_spaces = ct_spaces<t-1> + ct_string<2>(" ");
    template<> constexpr ct_string<1> ct_spaces<0> = ct_string<1>("");
    
    
    constexpr ct_string<2> to_ct_string( const bool b )
    {
        if( b )
        {
            return "1";
        }
        else
        {
            return "0";
        }
    }

    // Slows down everything?
    template<IntQ Int>
    constexpr ct_string<std::numeric_limits<Int>::digits10+3>
    to_ct_string( const Int i )
    {
        constexpr Size_T N = std::numeric_limits<Int>::digits10+3;

        constexpr char digits [11] = "0123456789";

        Int x [11];

        ct_string<N> s;

        Size_T x_ptr = 0;

        Int d = i % 10;
        Int r = i / 10;

        x[x_ptr++] = d;

        while( r != 0 )
        {
            d = r % 10;
            r = r / 10;

            x[x_ptr++] = d;
        }

        Int s_ptr = 0;

        if( i < Int(0) )
        {
            s[s_ptr++] = '-';
        }

        // Reverse
        while( x_ptr > Int(0) )
        {
            s[s_ptr++] = digits[x[--x_ptr]];
        }

        return s;
    }
} // namespace Tools
