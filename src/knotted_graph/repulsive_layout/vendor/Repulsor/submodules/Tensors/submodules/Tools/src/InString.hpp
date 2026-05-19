#pragma once

namespace Tools
{
    struct FromCharResult
    {
        const char * ptr = nullptr;
        bool failedQ = false;
    };
    
    template<typename T>
    struct FromChars
    {
        static constexpr bool implementedQ = false;
    };

    template<IntQ T>
    struct FromChars<T>
    {
        static constexpr bool implementedQ = true;
        
        FromCharResult operator()( const char * const begin, const char * const end, T & x ) const
        {
            auto r = std::from_chars( begin, end, x, 10 );
            return FromCharResult{ .ptr = r.ptr, .failedQ = (r.ec != std::errc{})};
        }
    };
    
    template<std::floating_point T>
    struct FromChars<T>
    {
        static constexpr bool implementedQ = true;
        
        FromCharResult operator()( const char * const begin, const char * const end, T & x ) const
        {
            auto r = std::from_chars( begin, end, x, std::chars_format::general );
            return FromCharResult{ .ptr = r.ptr, .failedQ = (r.ec != std::errc{})};
        }
    };
    
//    template<std::floating_point T>
//    struct ToChars<T>
//    {
//        static constexpr bool implementedQ = true;
//        
//        static Size_T CharCount( const T & x )
//        {
//            (void)x;
//            return std::numeric_limits<T>::max_digits10 + Size_T(7);
//        }
//        
//        ToCharResult operator()( char * & begin, char * end, const T & x ) const
//        {
//            auto r = std::to_chars( begin, end, x, std::chars_format::general, std::numeric_limits<T>::max_digits10 );
//            return ToCharResult{ .ptr = r.ptr, .failedQ = (r.ec == std::errc::value_too_large)};
//        }
//    };
    
//    template<std::floating_point T>
//    struct ToChars<std::complex<T>>
//    {
//        static constexpr bool implementedQ = true;
//        
//        static Size_T CharCount( const T & x )
//        {
//            (void)x;
//            return Size_T(2) * ToChars<T>::CharCount() + Size_T(2);
//        }
//        
//        ToCharResult operator()( char * & begin, char * end, const T & x ) const
//        {
//            if( begin + CharCount() > end )
//            {
//                return {.ptr = begin, .failedQ = true};
//            }
//            else
//            {
//                auto r = std::to_chars( begin, end, Re(x), std::chars_format::general, std::numeric_limits<T>::max_digits10 );
//                char * ptr = r.ptr;
//                *ptr = '+';
//                ++ptr;
//                r = std::to_chars( begin, end, Im(x), std::chars_format::general, std::numeric_limits<T>::max_digits10 );
//                *ptr = 'I';
//                ++ptr;
//                return {.ptr = begin, .failedQ = false};
//            }
//        }
//    };
    
    
    
    class InString
    {
        
    public:

        InString() = default;
        
        InString( const char * const begin_, const char * const end_ )
        :   begin { begin_ }
        ,   ptr   { begin_ }
        ,   end   { end_   }
        {}
        
        InString( const char * const begin_, Size_T size )
        :   InString{ begin_, &begin_[size] }
        {}
        
        explicit InString( std::string_view s )
        :   InString{ &*s.begin(), &*s.end() }
        {}

        explicit InString( cref<std::string> s )
        :   InString{ &*s.begin(), &*s.end() }
        {}
        
        explicit InString( std::string && s )
        :   buffer { std::move(s)     }
        ,   begin  { &*buffer.begin() }
        ,   ptr    { &*buffer.begin() }
        ,   end    { &*buffer.end()   }
        {}
        
        explicit InString ( cref<std::filesystem::path> file )
        {
            std::ifstream stream (file, std::ios::in | std::ios::binary);
            if( !stream )
            {
                eprint(ClassName() + "(std::filesystem::path): Opening file " + file.string() + " failed. Returning empty InString.");
                return;
            }
            
            // Obtain the size of the file.
            const auto file_size = std::filesystem::file_size(file);
            
            // Create a buffer.
            buffer = std::string(file_size, '\0');
            // Read the whole file into the buffer.
            stream.read(buffer.data(), static_cast<std::streamsize>(file_size));

            begin = &*buffer.begin();
            ptr   = &*buffer.begin();
            end   = &*buffer.end();
            
            if( FailedQ() )
            {
                eprint(ClassName() + "(std::filesystem::path): Reading from file " + file.string() + " failed.");
                return;
            }
        }
        
        ~InString() = default;
        
    private:
        
        std::string buffer;
        const char * begin  = nullptr;
        const char * ptr    = nullptr;
        const char * end    = nullptr;
        bool failedQ        = false;
        
    public:
        
        Size_T Size() const
        {
            return static_cast<Size_T>(std::distance(ptr,end));
        }
        
        Size_T Capacity() const
        {
            return static_cast<Size_T>(std::distance(begin,end));
        }
        
        bool FailedQ() const
        {
            return failedQ;
        }
        
        void SetFailedQ()
        {
            failedQ = true;
        }
        
        const char & operator[]( const Size_T i ) const
        {
            return ptr[i];
        }
        
        Size_T LineCount() const
        {
            Size_T line_count = 1;
            for( const char * c = ptr; c < end; ++c )
            {
                line_count += (*c == '\n');
            }
            return line_count;
        }
        
#include "InString/Take.hpp"
#include "InString/TakeMatrix.hpp"

//        static InString FromFile( cref<std::filesystem::path> file )
//        {
//            std::ifstream stream (file, std::ios::in | std::ios::binary);
//            if( !stream )
//            {
//                eprint(ClassName() + "(std::filesystem::path): Opening file " + file.string() + " failed. Returning empty InString.");
//                return InString();
//            }
//            
//            // Obtain the size of the file.
//            const auto file_size = std::filesystem::file_size(file);
//            
//            // Create a buffer.
//            std::string s = std::string(file_size, '\0');
//            // Read the whole file into the buffer.
//            stream.read(s.data(), static_cast<std::streamsize>(file_size));
//
//            return InString(std::move(s));
//        }

        
        template<bool checkQ = true>
        InString & Skip( const Size_T n )
        {
            ptr = std::min(ptr + n, end);   
            return *this;
        }
        
        InString & Pop()
        {
            if( ptr > begin )
            {
                --ptr;
            }
            else
            {
                eprint(MethodName("Pop") + ": String buffer is empty. Doing nothing.");
            }
            
            return *this;
        }
        
        InString & Pop( Size_T n )
        {
            if( ptr >= begin + n )
            {
                ptr -= n;
            }
            else
            {
                ptr = begin;
                eprint(MethodName("Pop") + ": Buffer size was smaller than n = " + ToString(n) + ". Emptying it completely.");
            }
            
            return *this;
        }
        
        bool EmptyQ() const
        {
            return ptr >= end;
        }
        
        char CurrentChar() const
        {
            return *ptr;
        }
        
        std::string_view View()
        {
            return std::string_view(ptr,end);
        }
        
    public:
        
        static std::string MethodName( const std::string & tag )
        {
            return ClassName() + "::" + tag;
        }
        
        static std::string ClassName()
        {
            return "InString";
        }
    };
    
} // Tools


