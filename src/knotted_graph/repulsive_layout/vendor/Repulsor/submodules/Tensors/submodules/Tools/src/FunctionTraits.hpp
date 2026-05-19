#pragma once

namespace Tools
{
    // See https://stackoverflow.com/a/7943765/8248900
    
    template<typename Lambda_T>
    struct function_traits : public function_traits<decltype(&Lambda_T::operator())>
    {};
    // For generic types, directly use the result of the signature of its 'operator()'
    
    template <typename Class_T, typename T, typename... Args>
    struct function_traits<T(Class_T::*)(Args...) const>
    // we specialize for pointers to member function
    {
//        enum { arity = sizeof...(Args) };
//        // arity is the number of arguments.
        
        using return_type = T;
        
        static constexpr std::size_t arity = sizeof...(Args); // arity is the number of arguments.
        
        template<std::size_t i>
        struct arg
        {
            using type = std::tuple_element<i,std::tuple<Args...>>::type;
            // the i-th argument is equivalent to the i-th tuple element of a tuple
            // composed of those arguments.
        };
    };
    
} // Tools
