#pragma once

#include <type_traits>


namespace mma
{
    template<typename T>
    constexpr bool HasTypeQ = (
        std::is_integral<T>::value
        || std::is_floating_point<T>::value
        || std::is_same<T,std::complex<double>>::value
        || std::is_same<T,std::complex<float>>::value
    );
    
    template< typename T, class = typename std::enable_if_t<HasTypeQ<T>>>
    using Type = std::conditional_t<
        std::is_integral<T>::value,
        mint,
        std::conditional_t<
            std::is_floating_point<T>::value,
            mreal,
            std::complex<mreal>
        >
    >;
} // namespace mma

