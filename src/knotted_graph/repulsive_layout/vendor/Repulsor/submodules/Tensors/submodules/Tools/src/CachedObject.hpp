#pragma once

#include <string>
#include <type_traits>
#include <any>
#include <unordered_map>

namespace Tools
{
    
    template<
        bool cacheQ_   = true , bool cache_mutexQ_   = true,
        bool p_cacheQ_ = false, bool p_cache_mutexQ_ = false, Size_T alignment = ObjectAlignment
    >
    class alignas(alignment) CachedObject
    {
    public:
        
        static constexpr bool cacheQ         = cacheQ_;
        static constexpr bool cache_mutexQ   = cacheQ && cache_mutexQ_;
        static constexpr bool p_cacheQ       = p_cacheQ_;
        static constexpr bool p_cache_mutexQ = p_cacheQ && p_cache_mutexQ_;
        
        struct FakeContainer_T {};
        
        using Container_T        = std::unordered_map<std::string,std::any>;
        using CacheContainer_T   = std::conditional_t<cacheQ,Container_T,FakeContainer_T>;
        using P_CacheContainer_T = std::conditional_t<p_cacheQ,Container_T,FakeContainer_T>;
        
        
        struct FakeMutex_T{};
        
        struct FakeLock_T
        {
            FakeLock_T( [[maybe_unused]] FakeMutex_T & mutex ) {}
            
            template<typename T>
            FakeLock_T( [[maybe_unused]] FakeMutex_T & mutex , [[maybe_unused]] T & anything ) {}
        };
        
        struct FakeGuard_T
        {
            FakeGuard_T( [[maybe_unused]] FakeMutex_T & mutex ) {}
        };
        
        using Mutex_T        = std::mutex;
        
        using CacheMutex_T   = std::conditional_t<cache_mutexQ,Mutex_T,FakeMutex_T>;
        using CacheLock_T    = std::unique_lock<Mutex_T>;
        using CacheGuard_T   = std::conditional_t<cache_mutexQ,std::lock_guard<Mutex_T>,FakeGuard_T>;
        
        using P_CacheMutex_T = std::conditional_t<p_cache_mutexQ,Mutex_T,FakeMutex_T>;
        using P_CacheLock_T  = std::unique_lock<Mutex_T>;
        using P_CacheGuard_T = std::conditional_t<cache_mutexQ,std::lock_guard<Mutex_T>,FakeGuard_T>;

    protected:
            
        [[no_unique_address]] mutable CacheContainer_T   cache;
        [[no_unique_address]] mutable P_CacheContainer_T p_cache;
        [[no_unique_address]] mutable CacheMutex_T       cache_mutex;
        [[no_unique_address]] mutable P_CacheMutex_T     p_cache_mutex;
        
    public:

        
        // Default constructor
        CachedObject() = default;
        
        // Destructor (virtual because this class is inherited by many others)
        virtual ~CachedObject() = default;
        
        // Copy constructor
        CachedObject( const CachedObject & other ) noexcept
        {
            if constexpr ( cacheQ )
            {
                if constexpr ( cache_mutexQ )
                {
                    CacheLock_T rhs_lk( other.cache_mutex );
                    this->cache = other.cache;
                }
                else
                {
                    this->cache = other.cache;
                }
            }
            
            if constexpr ( p_cacheQ )
            {
                if constexpr ( p_cache_mutexQ )
                {
                    P_CacheLock_T rhs_lk( other.p_cache_mutex );
                    this->p_cache = other.p_cache;
                }
                else
                {
                    this->p_cache = other.p_cache;
                }
            }
        }
        
        // Move constructor
        CachedObject( CachedObject && other) noexcept
        {
            if constexpr ( cacheQ )
            {
                if constexpr ( cache_mutexQ )
                {
                    CacheLock_T rhs_lk ( other.cache_mutex );
                    this->cache = std::move( other.cache   );
                }
                else
                {
                    this->cache = std::move( other.cache   );
                }
            }
            
            if constexpr ( p_cacheQ )
            {
                if constexpr ( p_cache_mutexQ )
                {
                    CacheLock_T rhs_lk ( other.p_cache_mutex );
                    this->p_cache = std::move( other.p_cache );
                }
                else
                {
                    this->p_cache = std::move( other.p_cache );
                }
            }
        }
        
        // Copy assignment
        CachedObject & operator=(const CachedObject & other ) noexcept
        {
            if( this != &other )
            {
                if constexpr ( cacheQ )
                {
                    if constexpr ( cache_mutexQ )
                    {
                        CacheLock_T lhs_lk( this->cache_mutex, std::defer_lock );
                        CacheLock_T rhs_lk( other.cache_mutex, std::defer_lock );
                        std::lock( lhs_lk, rhs_lk );
                        this->cache = other.cache;
                    }
                    else
                    {
                        this->cache = other.cache;
                    }
                }
                
                if constexpr ( p_cacheQ )
                {
                    if constexpr ( p_cache_mutexQ )
                    {
                        P_CacheLock_T lhs_lk( this->p_cache_mutex, std::defer_lock );
                        P_CacheLock_T rhs_lk( other.p_cache_mutex, std::defer_lock );
                        std::lock( lhs_lk, rhs_lk );
                        this->p_cache = other.p_cache;
                    }
                    else
                    {
                        this->p_cache = other.p_cache;
                    }
                }
            }
            return *this;
        }
        
        // Move assignment
        CachedObject & operator=( CachedObject && other ) noexcept
        {
            if (this != &other)
            {
                if constexpr ( cacheQ )
                {
                    if constexpr ( cache_mutexQ )
                    {
                        CacheLock_T lhs_lk ( this->cache_mutex, std::defer_lock );
                        CacheLock_T rhs_lk ( other.cache_mutex, std::defer_lock );
                        std::lock ( lhs_lk, rhs_lk );
                        this->cache = std::move( other.cache   );
                    }
                    else
                    {
                        this->cache = std::move( other.cache   );
                    }
                }
                
                if constexpr ( p_cacheQ )
                {
                    if constexpr ( p_cache_mutexQ )
                    {
                        P_CacheLock_T lhs_lk ( this->p_cache_mutex, std::defer_lock );
                        P_CacheLock_T rhs_lk ( other.p_cache_mutex, std::defer_lock );
                        std::lock ( lhs_lk, rhs_lk );
                        this->p_cache = std::move( other.p_cache   );
                    }
                    else
                    {
                        this->p_cache = std::move( other.p_cache   );
                    }
                }
            }
            return *this;
        }
        
        // Swap function
        friend void swap( CachedObject & x, CachedObject & y ) noexcept
        {
            if( &x != &y )
            {
                using std::swap;

                if constexpr ( cacheQ )
                {
                    if constexpr ( cache_mutexQ )
                    {
                        CacheLock_T lhs_lk( x.cache_mutex, std::defer_lock );
                        CacheLock_T rhs_lk( y.cache_mutex, std::defer_lock );
                        std::lock( lhs_lk, rhs_lk );
                        swap( x.cache, y.cache );
                    }
                    else
                    {
                        swap( x.cache, y.cache );
                    }
                }
                
                if constexpr ( p_cacheQ )
                {
                    if constexpr ( p_cache_mutexQ )
                    {
                        P_CacheLock_T lhs_lk( x.p_cache_mutex, std::defer_lock );
                        P_CacheLock_T rhs_lk( y.p_cache_mutex, std::defer_lock );
                        std::lock( lhs_lk, rhs_lk );
                        swap( x.p_cache, y.p_cache );
                    }
                    else
                    {
                        swap( x.p_cache, y.p_cache );
                    }
                }
            }
        }
        
    public:
        
//###########################################################
//      Cache
//###########################################################
        
        template<bool lockQ = true, typename dummy = void>
        bool InCacheQ( cref<std::string> key ) const
        {
            if constexpr ( cacheQ )
            {
                if constexpr( lockQ )
                {
                    const CacheGuard_T cache_lock( cache_mutex );
                    return cache.contains( key );
                }
                else
                {
                    // The version without lock is necessary because we call InCacheQ frequently from locked routines.
                    return cache.contains( key );
                }
            }
            else
            {
                static_assert(DependentFalse<dummy>,"We should not arrive here");
            }
            
            return false;
        }
        
        template<typename T = std::any, typename dummy = void>
        T & GetCache( cref<std::string> key ) const
        {
            // For some reason gcc-12 does not allow me to return cache.at(key) directly. =/
            // Maybe because that might throw an exception.
            
            std::any * thing;
            
            if constexpr ( cacheQ )
            {
                const CacheGuard_T cache_lock( cache_mutex );
                
                try
                {
                    thing = &cache.at( key );
                }
                catch( const std::out_of_range & e )
                {
                    eprint(this->ClassName()+"::GetCache: Key \""+key+"\" not found!.");
                    throw(e); //an internal catch block forwards the exception to its external level
                }
                
                if constexpr ( std::is_same_v<T,std::any> )
                {
                    return *thing;
                }
                else
                {
                    return std::any_cast<T &>(*thing);
                }
            }
            else
            {
                static_assert(DependentFalse<dummy>,"We should not arrive here");
            }
        }
        
//        // Caution! This function is destructive.
//        void SetCache( cref<std::string> key, std::any & thing ) const
//        {
//            const std::lock_guard<std::mutex> cache_lock( cache_mutex );
//            cache[key] = std::move(thing);
//        }
        
        // Caution! This function is destructive.
        template<bool check_existenceQ = true, typename dummy = void>
        void SetCache( cref<std::string> key, std::any && thing ) const
        {
            if constexpr ( cacheQ )
            {
                const CacheGuard_T cache_lock( cache_mutex );
                
                if constexpr ( check_existenceQ )
                {
                    if( this->template InCacheQ<false>(key) )
                    {
                        wprint( this->ClassName()+"::SetCache: Key \"" + key + "\" is already in cache. Maybe check for race conditions." );
                    }
                }
                
                cache[key] = std::move(thing);
            }
            else
            {
                static_assert(DependentFalse<dummy>,"We should not arrive here");
            }
        }
        
        std::string CacheKeys() const
        {
            std::string s;
            
            if constexpr ( cacheQ )
            {
                const CacheGuard_T cache_lock( cache_mutex );
                
                Size_T iter = 0;
                s += "{ ";
                for( auto const & p : cache )
                {
                    if( iter > Size_T(0) ) { s += ", "; }
                    s += p.first;
                    ++iter;
                }
                s += " }";
            }
            
            return s;
        }
        
        void ClearCache() const
        {
            if constexpr ( cacheQ )
            {
                const CacheGuard_T cache_lock( cache_mutex );
                
                if( !cache.empty() )
                {
                    cache.clear();
                }
            }
        }
        
        void ClearCache( const std::string & key ) const
        {
            if constexpr ( cacheQ )
            {
                const CacheGuard_T cache_lock( cache_mutex );
                
                if( this->template InCacheQ<false>( key ) )
                {
                    cache.erase( key );
                }
            }
        }
        
        
//###########################################################
//      PersistentCache
//###########################################################
        
        template<bool lockQ = true, typename dummy = void>
        bool InPersistentCacheQ( cref<std::string> key ) const
        {
            bool result = false;
            
            if constexpr ( p_cacheQ )
            {
                if constexpr( lockQ )
                {
                    if constexpr( lockQ )
                    {
                        const P_CacheGuard_T p_cache_lock( p_cache_mutex );
                        result = p_cache.contains( key );
                    }
                    else
                    {
                        result = p_cache.contains( key );
                    }
                }
            }
            else
            {
                static_assert(DependentFalse<dummy>,"We should not arrive here");
            }
            
            return result;
        }
        
        template<typename T = std::any, typename dummy = void>
        T & GetPersistentCache( cref<std::string> key ) const
        {
            if constexpr ( p_cacheQ )
            {
                std::any * thing;
                
                const P_CacheGuard_T p_cache_lock( p_cache_mutex );
                
                try
                {
                    thing = &p_cache.at( key );
                }
                catch( const std::out_of_range & e )
                {
                    eprint(this->ClassName()+"GetPersistentCache: Key \""+key+"\" not found!.");
                    throw(e);
                }
                
                if constexpr ( std::is_same_v<T,std::any> )
                {
                    return *thing;
                }
                else
                {
                    return std::any_cast<T &>(*thing);
                }
            }
            else
            {
                static_assert(DependentFalse<dummy>,"We should not arrive here");
            }
        }
        
        // Caution! This function is destructive.
        template<bool check_existenceQ = true, typename dummy = void>
        void SetPersistentCache( cref<std::string> key, std::any && thing ) const
        {
            if constexpr ( p_cacheQ )
            {
                const P_CacheGuard_T p_cache_lock( p_cache_mutex );
                
                if constexpr ( check_existenceQ )
                {
                    if( this->template InPersistentCacheQ<false>(key) )
                    {
                        wprint( this->ClassName()+"::SetPersistentCache: Key \"" + key + "\" is already in cache. Maybe check for race conditions." );
                    }
                }
                
                p_cache[key] = std::move(thing);
            }
            else
            {
                static_assert(DependentFalse<dummy>,"We should not arrive here");
            }
        }
        
        std::string PersistentCacheKeys() const
        {
            std::string s;
            
            if constexpr ( p_cacheQ )
            {
                const P_CacheGuard_T p_cache_lock( p_cache_mutex );

                s += "{\n";
                for( auto const & p : p_cache )
                {
                    s += "\t";
                    s += p.first;
                    s += "\n";
                }
                s += "}";
            }
            
            return s;
        }
        
        void ClearPersistentCache() const
        {
            if constexpr ( p_cacheQ )
            {
                const P_CacheGuard_T p_cache_lock( p_cache_mutex );
                
                if( !p_cache.empty() )
                {
                    p_cache.clear();
                }
            }
        }
        
        void ClearPersistentCache( cref<std::string> key ) const
        {
            if constexpr ( p_cacheQ )
            {
                const P_CacheGuard_T p_cache_lock( p_cache_mutex );
                
                if( this->template InPersistentCacheQ<false>( key ) )
                {
                    p_cache.erase( key );
                }
            }
        }
        
        
        
//###########################################################
//      Misc
//###########################################################
        
        void ClearAllCache() const
        {
            if constexpr ( cacheQ )
            {
                const CacheGuard_T cache_lock( cache_mutex );
                
                cache.clear();
            }
            
            if constexpr ( p_cacheQ )
            {
                const P_CacheGuard_T p_cache_lock( p_cache_mutex );
                
                p_cache.clear();
            }
        }
        
        
        static auto ClassName()
        {
            return ct_string("CachedObject");
        }
    
    }; // CachedObject
    
} // namespace Tools
