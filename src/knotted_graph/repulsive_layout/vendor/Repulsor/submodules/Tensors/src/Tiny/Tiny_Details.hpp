public:

#include "Tiny_Constants.hpp"

    template<class T>
    TOOLS_FORCE_INLINE mref<Class_T> operator/=( const T lambda )
    {
        return (*this) *= ( scalar_cast<Scal>(Inv<T>(lambda)) );
    }
