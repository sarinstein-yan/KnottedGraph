#pragma once

namespace Tools
{
    static std::mutex cout_mutex;
    static std::mutex cerr_mutex;
    
    // We need it here because print might need knowledge of it when running in Mathematica.
    using Clock = std::chrono::high_resolution_clock;
    using Time  = std::chrono::time_point<Clock>;
    
    namespace Timer
    {
        static std::vector<Time> stack;
        static std::mutex mutex;
    }
    
    inline void print( std::string_view s )
    {
        const std::lock_guard<std::mutex> cout_lock( Tools::cout_mutex  );
        
#if defined(LTEMPLATE_H) || defined(TENSORS_MMA_HPP)
        const std::lock_guard<std::mutex> timer_lock( Timer::mutex );
        const Size_T len = 2 * (Timer::stack.size());
        std::string s_out (len + s.size(), ' ');
        std::copy_n( &s[0], s.size(), &s_out[len] );
        mma::print(s_out);
#else
        std::cout << s << std::endl;
#endif
    }
    
    template<Size_T n>
    inline void print( const char(&c)[n]  )
    {
        print(std::string_view(&c[0],&c[n-1]));
    }
    
    template<Size_T N>
    inline void print( const ct_string<N> & s )
    {
        print( s.data() );
    }
    
//    template<Size_T align = 0, typename T>
//    inline void valprint( const std::string & s, const T & value)
//    {
//        const Size_T len = (align > Size_T(0)) ? std::max(s.size(),align) : s.size();
//        
//        std::string s_out (len + 3, ' ');
//        std::copy_n( s.begin(), s.size(), s_out.begin() );
//        s_out[len+1] = '=';
//        s_out += COND(FloatQ<T>,ToString(value),ToString(value));
//        print(s_out);
//    }
//    
//    template<Size_T align = 0, typename T>
//    inline void valprint( const std::string & s, const std::string & value)
//    {
//        const Size_T len = (align > Size_T(0)) ? std::max(s.size(),align) : s.size();
//        
//        std::string s_out (len + 3 + value.size(),' ');
//        std::copy_n( s.begin(), s.size(), s_out.begin() );
//        s_out[len+1] = '=';
//        std::copy_n( value.begin(), value.size(), &s_out[len + 3] );
//        print(s_out);
//    }
    
    template<Size_T align = 0, typename T>
    inline void valprint( std::string_view s, const T & value)
    {
        const Size_T len = (align > Size_T(0)) ? std::max(s.size(),align) : s.size();
        
        std::string s_out (len + 3, ' ');
        std::copy_n( s.begin(), s.size(), s_out.begin() );
        s_out[len+1] = '=';
        s_out += ToString(value);
        print(s_out);
        
//        std::string s_out (s);
//        s_out += " = ";
//        s_out += ToString(value);
//        print(s_out);
    }
    
    template<Size_T align = 0>
    inline void valprint( std::string_view s, std::string_view value)
    {
        const Size_T len = (align > Size_T(0)) ? std::max(s.size(),align) : s.size();
        
        std::string s_out (len + 3 + value.size(),' ');
        std::copy_n( s.begin(), s.size(), s_out.begin() );
        s_out[len+1] = '=';
        std::copy_n( value.begin(), value.size(), &s_out[len + 3] );
        print(s_out);
        
//        std::string s_out (s);
//        s_out += " = ";
//        s_out += value;
//        print(s_out);
    }

} // namespace Tools


#define TOOLS_DUMP(x) Tools::valprint( std::string_view(#x), x )

#define TOOLS_MEM_DUMP_STRING(x) "\"" + std::string(#x) + "\" -> " + Tools::ToMathematicaString( static_cast<double>((x).AllocatedByteCount()) )

#define TOOLS_MEM_DUMP(x) Tools::print( TOOLS_MEM_DUMP_STRING(x) )
