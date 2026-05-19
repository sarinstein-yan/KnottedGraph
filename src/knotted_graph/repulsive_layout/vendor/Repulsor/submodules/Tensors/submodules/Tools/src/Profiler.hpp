#pragma once

#include <mutex>

#include <filesystem>

namespace Tools
{
    
    /*!@brief Return user's home directory.
     */
    
    static std::filesystem::path HomeDirectory(void)
    {
        const char * s = std::getenv("HOME");

        if( s == nullptr )
        {
            std::filesystem::path home_dir {std::getenv("HOMEDRIVE")};
            
            home_dir /= std::getenv("HOMEPATH");
            
            return home_dir;
        }
        else
        {
            return std::filesystem::path { s };
        }
    }
    
    namespace Profiler
    {
        using Int = std::int64_t;
        
        static std::filesystem::path log_file  = HomeDirectory() / "Tools_Log.txt";
        static std::filesystem::path prof_file = HomeDirectory() / "Tools_Profile.tsv";
        
        static std::mutex log_mutex;
        static std::ofstream log  ( log_file );
        
        // Only used in builds with TOOLS_ENABLE_PROFILER.
        [[maybe_unused]] static int blocker_count = 0;
        
        struct StackNode
        {
            Int id;
            Int parent;
            std::string tag;
            Time time;
            
            StackNode(
                Int id_,
                Int parent_,
                const std::string & tag_
            )
            :   id    ( id_     )
            ,   parent( parent_ )
            ,   tag   ( tag_    )
            ,   time  ( Tools::Clock::now() )
            {}
        };
        
        struct Blocker
        {
            Blocker()
            {
#if defined(TOOLS_ENABLE_PROFILER)
                ++blocker_count;
#endif
            }
            
            ~Blocker()
            {
#if defined(TOOLS_ENABLE_PROFILER)
                --blocker_count;
#endif
            }
        };
        
#if defined(TOOLS_ENABLE_PROFILER)
        static std::mutex prof_mutex;
        static std::ofstream prof ( prof_file );
        
        static Int id_counter = 0;
        
        static std::vector<StackNode> stack (
            Size_T(1), StackNode( id_counter++, Int(-1), "root" )
        );

#endif // defined(TOOLS_ENABLE_PROFILER)
        
        
        inline void Clear(
            const std::filesystem::path & dir,
            const bool silentQ = false,
            const bool appendQ = false
        )
        {
            const std::lock_guard<std::mutex> log_lock ( log_mutex  );
            
            Profiler::log_file = dir / "Tools_Log.txt";
            
            Profiler::log.close();

            if( appendQ )
            {
                Profiler::log.open( Profiler::log_file, std::ios_base::app );
            }
            else
            {
                Profiler::log.open( Profiler::log_file );
            }
            
            if( Profiler::log.good() )
            {
                if( !silentQ )
                {
                    print( std::string("Log     will be written to ") + Profiler::log_file.string() + "." );
                }
            }
            else
            {
                print( std::string("ERROR: Profiler failed to open file ") + Profiler::log_file.string() + "." );
            }

            Profiler::log << std::setprecision(16);
            
#if defined(TOOLS_ENABLE_PROFILER)
            const std::lock_guard<std::mutex> prof_lock( prof_mutex );
            
            Profiler::prof_file = dir / "Tools_Profile.tsv";
            
            Profiler::prof.close();
            
            if( appendQ )
            {
                Profiler::prof.open( Profiler::prof_file, std::ios_base::app );
            }
            else
            {
                Profiler::prof.open( Profiler::prof_file );
            }

            if( Profiler::prof.good() )
            {
                if( !silentQ )
                {
                    print( std::string("Profile will be written to ") + Profiler::prof_file.string() + ".");
                }
            }
            else
            {
                print( std::string("ERROR: Profiler failed to open file ") + Profiler::prof_file.string() + "." );
            }
            
            id_counter = 0;
            stack.clear();
            stack.emplace_back(id_counter++,Int(-1),"root");
#endif // defined(TOOLS_ENABLE_PROFILER)
        }
        
        inline void Clear( const bool silentQ = false, const bool appendQ = false )
        {
            Clear( HomeDirectory(), silentQ, appendQ );
        }
        
        
        template<bool tabsQ = true>
        inline void logprint( std::string_view s )
        {
            const std::lock_guard<std::mutex> lock( log_mutex );
            if constexpr ( tabsQ )
            {
#if defined(TOOLS_ENABLE_PROFILER)
                for( Size_T i = 0; i < 2 * (Profiler::stack.size()+1); ++i )
                {
                    Profiler::log <<  ' ';
                }
//                Profiler::log << std::string( 2 * (Profiler::stack.size()+1), ' ' );
#endif
            }
            Profiler::log << s << "\n" << std::endl;
        }
        
        template<bool tabsQ = true, Size_T N>
        inline void logprint( const ct_string<N> & s )
        {
            logprint<tabsQ>( s.data() );
        }
        
        
        template<bool tabsQ = true, typename T>
        inline void logvalprint( std::string_view tag, const T & value )
        {
            const std::lock_guard<std::mutex> lock( log_mutex );
            if constexpr ( tabsQ )
            {
#if defined(TOOLS_ENABLE_PROFILER)
                for( Size_T i = 0; i < 2 * (Profiler::stack.size()+1); ++i )
                {
                    Profiler::log <<  ' ';
                }
//                Profiler::log << std::string( 2 * (Profiler::stack.size()+1), ' ' );
#endif
            }
            Profiler::log << tag << " = " << ToString(value) << "\n" << std::endl;
        }
        
        inline void eprint( std::string_view s )
        {
            std::string msg ("ERROR: ");
            msg += s;
            
    #if defined(LTEMPLATE_H) || defined(TENSORS_MMA_HPP)
            print( msg );
    #endif
            const std::lock_guard<std::mutex> cerr_lock( Tools::cerr_mutex );
            std::cerr << msg << std::endl;
            logprint<false>( msg );
        }
        
        template<Size_T N>
        inline void eprint( const ct_string<N> & s )
        {
            eprint( s.data() );
        }
        
        
        inline void error( std::string_view s )
        {
            std::string msg ("ERROR: ");
            msg += s;
            eprint(msg);
            throw std::runtime_error(msg);
        }
        
        template<Size_T N>
        inline void error( const ct_string<N> & s )
        {
            error( s.data() );
        }
        
        
        inline void Tic( const std::string & tag)
        {
    #ifdef TOOLS_ENABLE_PROFILER
            
            if( blocker_count > 0 ) { return; }
                
            const std::lock_guard<std::mutex> prof_lock( prof_mutex );
            
            stack.emplace_back( id_counter++, stack.back().id, tag );
            
            double start_time = Tools::Duration( stack[0].time, stack.back().time );
            
            const std::lock_guard<std::mutex> log_lock( log_mutex );
            
            log
                << std::string( 2 * (stack.size()), ' ' )
                << tag << "\t started at \t" << start_time << "\n"
                << std::endl;
    #else
            (void)tag;
    #endif
        }
        
        template<Size_T N>
        inline void Tic( const ct_string<N> & tag )
        {
    #ifdef TOOLS_ENABLE_PROFILER
            Tic(tag.data());
    #else
            (void)tag;
    #endif
        }
        
        inline void Toc( const std::string & tag)
        {
    #ifdef TOOLS_ENABLE_PROFILER
            
            if( blocker_count > 0 ) { return; }
            
            const std::lock_guard<std::mutex> prof_lock( prof_mutex );
            
            if( !stack.empty() )
            {
                StackNode & node = stack.back();
                
                if( tag == node.tag )
                {
                    Time & init_time  = stack[0].time;
                    double start_time = Tools::Duration( init_time, node.time );
                    double stop_time  = Tools::Duration( init_time, Clock::now() );
                    
                    prof
                        << node.id <<  "\t"
                        << node.tag << "\t"
                        << node.parent << "\t"
                        << start_time << "\t"
                        << stop_time << "\t"
                        << stop_time-start_time << "\t"
                        << stack.size()-1
                        << std::endl;
                    
                    const std::lock_guard<std::mutex> log_lock( log_mutex );
                    log
                        << std::string( 2 * (stack.size()), ' ' )
                        << node.tag << "\t ended   at \t" << stop_time << "\n"
                        << std::endl;
                    
                    stack.pop_back();
                }
                else
                {
                    eprint( std::string("Unmatched TOOLS_PTOC detected. Tag requested = ") + tag + ". Tag found = " + node.tag + ".");
                }
            }
            else
            {
                eprint( std::string("Unmatched TOOLS_PTOC detected. Stack empty. Label = ") + tag + ".");
            }
    #else
            (void)tag;
    #endif
        }
        
        template<Size_T N>
        inline void Toc( const ct_string<N> & tag )
        {
    #ifdef TOOLS_ENABLE_PROFILER
            Toc(tag.data());
    #else
            (void)tag;
    #endif
        }
        
        class Timer final
        {
        public:
            
            const std::string tag;
            bool activeQ = false;
            
        public:
            
#ifdef TOOLS_ENABLE_PROFILER
            Timer( std::string_view tag_ )
            :   tag { tag_ }
            ,   activeQ( blocker_count <= 0 )
            {
                if( activeQ ) { Tic(tag); };
            }
//            
//            Timer( std::string_view && tag_ )
//            :   tag { std::move(tag_) }
//            ,   activeQ( blocker_count <= 0 )
//            {
//                if( activeQ ) { Tic(tag); };
//            }

            ~Timer()
            {
                if( activeQ ) { Toc(tag); };
            }
#else
            Timer( std::string_view tag_ )
            {

                (void)tag_;
            }
//            
//            Timer( std::string && tag_ )
//            {
//                (void)tag_;
//            }
#endif
        };
        
    } // namespace Profiler
    
    
// Print functions that are always active and print to the log file.
    
    /*!
     * @brief Print to log file specified in `Profiler::log_file`.
     */
    
    template<bool tabsQ = true>
    inline void logprint( std::string_view s )
    {
        Profiler::logprint<tabsQ>(s);
    }
    
    template<bool tabsQ = true, Size_T N>
    inline void logprint( const ct_string<N> & s )
    {
        Profiler::logprint<tabsQ>(s);
    }
    
    
    /*!@brief Print an error to `std::cout` and `std::cerr`.
     */
    
    inline void eprint( std::string_view s )
    {
        Profiler::eprint(s);
    }
    
    template<Size_T N>
    inline void eprint( const ct_string<N> & s )
    {
        Profiler::eprint(s);
    }
    
    /*!@brief Print log message and throw a `std::runtime_error`.
     */
    
    inline void error( std::string_view s )
    {
        Profiler::error(s);
    }
    
    template<Size_T N>
    inline void error( const ct_string<N> & s )
    {
        Profiler::error(s);
    }

    
    /*!
     * @brief Print to log file specified in `Profiler::log_file`.
     */
    
    template<typename T>
    inline void logvalprint( std::string_view s, const T & value )
    {
//        logprint( s + " = " + ToString(value) );
        Profiler::logvalprint(s,value);
    }
    
    template<Size_T N, typename T>
    inline void logvalprint( const ct_string<N> & s, const T & value )
    {
        Profiler::logvalprint( s.data(), value );
    }

//    /*!
//     * @brief Print to log file specified in `Profiler::log_file`.
//     */
//    
//    template<typename T>
//    inline void logvalprint( const std::string & s, const T & value, const int p )
//    {
//        Profiler::logvalprint( s, ToString(value, p) );
//    }
//    
//    template<Size_T N, typename T>
//    inline void logvalprint( const ct_string<N> & s, const T & value, const int p )
//    {
//        Profiler::logvalprint( s.data(), value, p  );
//    }
    
//    /*!
//     * @brief Print to log file specified in `Profiler::log_file`.
//     */
//    
//    inline void logvalprint( const std::string & s, const std::string & value )
//    {
//        Profiler::logvalprint( s, value );
//    }
//    
//    template<Size_T N>
//    inline void logvalprint( const ct_string<N> & s, const std::string & value )
//    {
//        Profiler::logvalprint( s.data(), value  );
//    }
    
    
#define TOOLS_LOGDUMP(x) Tools::logvalprint( std::string_view(#x), x );
    
#define TOOLS_DDUMP(x) Tools::logvalprint( std::string_view(#x), x ); Tools::valprint( std::string_view(#x), x );
    

    
    /*!@brief Print a warning to `std::cout`.
     */
    
    inline void wprint( std::string_view s )
    {
        std::string msg ("WARNING: ");
        msg += s;
#if defined(LTEMPLATE_H) || defined(TENSORS_MMA_HPP)
        print( msg );
#endif
        const std::lock_guard<std::mutex> cerr_lock( Tools::cerr_mutex );
        std::cerr << msg << std::endl;
        logprint<false>(msg);
    }
    
    template<Size_T N>
    inline void wprint( const ct_string<N> & s )
    {
        wprint(s.data());
    }
    
    /*!
     * @brief Print a note to `std::cout`.
     */
    
    inline void nprint( std::string_view s )
    {
        std::string msg ("NOTE: ");
        msg += s;
#if defined(LTEMPLATE_H) || defined(TENSORS_MMA_HPP)
        print( msg );
#endif
        const std::lock_guard<std::mutex> cerr_lock( Tools::cerr_mutex );
        std::cerr << msg << std::endl;
        logprint<false>(msg);
    }
    
    template<Size_T N>
    inline void nprint( const ct_string<N> & s )
    {
        nprint(s.data());
    }
    
    /*!@brief This print function is only active if macro `TOOLS_ENABLE_PROFILER` is defined.
     * It prints to the log file specified in `Profiler::log_file`.
     */
    
    inline void pprint( std::string_view s )
    {
#ifdef TOOLS_ENABLE_PROFILER
        const std::lock_guard<std::mutex> lock( Profiler::log_mutex );
        Profiler::log << s << "\n" << std::endl;
#else
        (void)s;
#endif
    }

    /*!@brief This print function is only active if macro `TOOLS_ENABLE_PROFILER` is defined.
     * It prints to the log file specified in `Profiler::log_file`.
     */
    
    template<typename T>
    inline void pvalprint( std::string_view s, const T & value)
    {
#ifdef TOOLS_ENABLE_PROFILER
        const std::lock_guard<std::mutex> lock( Profiler::log_mutex );
        Profiler::log << s << " = " << ToString(value) << "\n" << std::endl;
#else
        (void)s;
        (void)value;
#endif
    }

    /*!@brief This print function is only active if macro `TOOLS_ENABLE_PROFILER` is defined.
     * It prints to the log file specified in `Profiler::log_file`.
     */
    
    template<typename T>
    inline void pvalprint( std::string_view s, const T & value, const int p)
    {
#ifdef TOOLS_ENABLE_PROFILER
        Profiler::log << s << " = " << ToString(value,p) << "\n" << std::endl;
#else
        (void)s;
        (void)value;
        (void)p;
#endif
    }
    
    /*!@brief This print function is only active if macro `TOOLS_ENABLE_PROFILER` is defined.
     * It prints to the log file specified in `Profiler::log_file`.
     */
    
    inline void pvalprint( std::string_view s, std::string_view value)
    {
#ifdef TOOLS_ENABLE_PROFILER
        Profiler::log << s << " = " << value << "\n" << std::endl;
#else
        (void)s;
        (void)value;
#endif
    }
    
#define TOOLS_PDUMP(x) pvalprint( std::string_view(#x), x );

    
#ifdef TOOLS_ENABLE_PROFILER
    #define TOOLS_PTIC( s ) Tools::Profiler::Tic(s)
    #define TOOLS_PTOC( s ) Tools::Profiler::Toc(s)
    #define TOOLS_PTIMER( name, s ) Tools::Profiler::Timer name (s);
#else
    #define TOOLS_PTIC( s )
    #define TOOLS_PTOC( s )
#define TOOLS_PTIMER( name, s )
#endif
    
    
#ifdef TOOLS_DEBUG
    #define TOOLS_DEBUG_TIC(tag) TOOLS_PTIC(tag);
#else
    #define TOOLS_DEBUG_TIC(tag)
#endif
    
#ifdef TOOLS_DEBUG
    #define TOOLS_DEBUG_TOC(tag) TOOLS_PTOC(tag);
#else
    #define TOOLS_DEBUG_TOC(tag)
#endif
    
#ifdef TOOLS_DEBUG
    #define TOOLS_DEBUG_PRINT(s) logprint(s);
#else
    #define TOOLS_DEBUG_PRINT(s)
#endif
    
#ifdef TOOLS_DEBUG
    #define TOOLS_DEBUG_ASSERT(condition, s) if( ! condition ) { eprint(s); }
#else
    #define TOOLS_DEBUG_ASSERT(condition, s)
#endif

} // namespace Tools
