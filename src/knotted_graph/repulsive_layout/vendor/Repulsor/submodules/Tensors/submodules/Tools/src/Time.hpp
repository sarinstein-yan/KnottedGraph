#pragma once

namespace Tools
{
    TOOLS_FORCE_INLINE double Duration(
        const Time & start_time, const Time & stop_time
    )
    {
        return std::chrono::duration<double>(stop_time - start_time).count();
    }
    
    class TimePoint
    {
        std::chrono::time_point<Clock> t;
        
    public:
        
        friend class TimeInterval;
        
        TimePoint( bool flag )
        {
            (void)flag;
            
            Now();
        }
        
        TimePoint() = default;
        
        void Now()
        {
            t = Clock::now();
        }
        
        inline friend double Duration( cref<TimePoint> a, cref<TimePoint> b )
        {
            return std::chrono::duration<double>(b.t - a.t).count();
        }
    };
    
    class TimeInterval final
    {
        TimePoint t [2];
        
    public:
        
        
        TimeInterval( bool flag )
        {
            (void)flag;
            Tic();
        }
        
        TimeInterval() = default;
        
        ~TimeInterval() = default;
        
        
        cref<TimePoint> operator[]( bool i )
        {
            return t[i];
        }
        
        template<bool do_itQ = true>
        void Tic()
        {
            if constexpr ( do_itQ )
            {
                t[0].Now();
            }
        }
        
        template<bool do_itQ = true>
        void Toc()
        {
            if constexpr ( do_itQ )
            {
                t[1].Now();
            }
        }
        
        double Duration() const
        {
            return std::chrono::duration<double>(t[1].t - t[0].t).count();
        }
    };

    inline void tic(const std::string & s)
    {
        print( s + "..." );
        {
            const std::lock_guard<std::mutex> timer_lock( Timer::mutex );
            
            Timer::stack.push_back(Clock::now());
        }
    }

    inline double toc(const std::string & s)
    {
        double duration = 0;
        
        if (!Timer::stack.empty())
        {
            {
                const std::lock_guard<std::mutex> timer_lock( Timer::mutex );
                duration = Tools::Duration( Timer::stack.back(), Clock::now() );
                Timer::stack.pop_back();
            }
            print( std::to_string(duration) + " s.");
        }
        else
        {
            print("Unmatched toc detected. Label =  " + s);
        }
        
        return duration;
    }
    
} // namespace Tools
