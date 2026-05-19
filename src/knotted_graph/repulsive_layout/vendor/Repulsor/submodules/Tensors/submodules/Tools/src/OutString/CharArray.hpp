#pragma once

namespace Tools
{
    template<Size_T n>
    class CharArray
    {
        public:
        
        constexpr CharArray( const char(&c)[n] )
        {
            std::copy_n(&c[0],n,&a[0]);
        };
        
    private:
        
        std::array<char,n> a;
        
    public:
        
        static constexpr Size_T Size()
        {
            return n-1;
        }
      
        constexpr char * begin()
        {
            return &a[0];
        }
        
        constexpr const char * begin() const
        {
            return &a[0];
        }
        
        constexpr char * end()
        {
            return &a[n-1];
        }
        
        constexpr const char * end() const
        {
            return &a[n-1];
        }
        
        constexpr char & operator[]( const Size_T i )
        {
            return a[i];
        }
        
        constexpr const char & operator[]( const Size_T i ) const
        {
            return a[i];
        }
        
        
        constexpr ToCharResult ToChars( char * begin, const char * end )
        {
            if( begin + Size() <= end )
            {
                std::copy_n( &a[0], Size(), begin );
                return { .ptr = begin + Size(), .failedQ = false };
            }
            else
            {
                return { .ptr = begin, .failedQ = true };
            }
        }
        
        constexpr friend Size_T CharCount( const CharArray & char_array )
        {
            (void)char_array;
            return Size();
        }
    };


    
} // namespace Tools
