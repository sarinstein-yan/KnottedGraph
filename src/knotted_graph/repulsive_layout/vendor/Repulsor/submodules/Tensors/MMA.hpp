#pragma once

//#define MATHEMATICA

#define TENSORS_MMA_HPP

#include "mathlink.h"

#ifdef True
    #undef True
#endif

#ifdef False
    #undef False
#endif

// Unbelievable that they defined a function-like macro with the name `P`...
#if defined( P )
    #undef P
#endif

#if defined( FAR )
    #undef FAR
#endif

#if defined( E )
    #undef E
#endif

#if defined( Pi )
    #undef Pi
#endif


#include <string>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <complex>

#include "mathlink.h"
#include "WolframLibrary.h"
#include "WolframSparseLibrary.h"

using Real    = mreal;
using Complex = std::complex<Real>;
using Int     = mint;

#include "src/MathematicaTypes.hpp"

namespace mma
{
    WolframLibraryData libData;
    
    inline void print(const char *msg)
    {
        if (libData->AbortQ())
        {
            return; // trying to use the MathLink connection during an abort appears to break it
        }

        MLINK link = libData->getMathLink(libData);
        
        MLPutFunction(link, "EvaluatePacket", 1);
        
        MLPutFunction(link, "Print", 1);
        
        MLPutString(link, msg);
        
        libData->processMathLink(link);
        
        int pkt = MLNextPacket(link);
        
        if (pkt == RETURNPKT)
        {
            MLNewPacket(link);
        }
    }

    // Call _Mathematica_'s `Print[]`, `std::string` argument version.
    inline void print(const std::string &msg)
    {
        print(msg.c_str());
    }
    
    
    template<typename Scal> inline Scal & get( MArgument marg );
    
    template<> inline Int & get<Int>( MArgument marg )
    {
        return *((marg).integer);
    }
    
    template<> inline Real & get<Real>( MArgument marg )
    {
        return *((marg).real);
    }
    
    template<> inline mcomplex & get<mcomplex>( MArgument marg )
    {
        return *((marg).cmplex);
    }
    
    template<> inline Complex & get<Complex>( MArgument marg )
    {
        return *(reinterpret_cast<Complex*>((marg).cmplex));
    }
    
    template<> inline MTensor & get<MTensor>( MArgument marg )
    {
        return *((marg).tensor);
    }
    
    template<> inline MSparseArray & get<MSparseArray>( MArgument marg )
    {
        return *((marg).sparse);
    }
    
    template<> inline mbool & get<mbool>( MArgument marg )
    {
        return *((marg).boolean);
    }
    
    template<> inline char* & get<char*>( MArgument marg )
    {
        return *((marg).utf8string);
    }
    
    
    template<typename Scal> inline Scal * data( MTensor & M );
    
    template<> inline Real * data<Real>( MTensor & M )
    {
        return libData->MTensor_getRealData(M);
    }
    
    template<> inline Int * data<Int>( MTensor & M )
    {
        return libData->MTensor_getIntegerData(M);
    }
    
    template<> inline mcomplex * data<mcomplex>( MTensor & M )
    {
        return libData->MTensor_getComplexData(M);
    }
    
    template<> inline Complex * data<Complex>( MTensor & M )
    {
        return reinterpret_cast<Complex *>( libData->MTensor_getComplexData(M) );
    }
    
    
    
    inline const Int * dimensions( MTensor & M )
    {
        return libData->MTensor_getDimensions(M);
    }
    
    inline Int rank( MTensor & M )
    {
        return libData->MTensor_getRank(M);
    }
    
    
    template<typename Scal> inline Scal * data( MArgument marg )
    {
        return data<Scal>(get<MTensor>(marg));
    }
    
    
    template<typename Scal>
    inline MTensor make_MTensor( const Int rank, const Int * dims );
    
    template<> inline MTensor make_MTensor<Int>( const Int rank, const Int * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Integer, rank, dims, &M );
        
        return M;
    }
    
    template<> inline MTensor make_MTensor<Real>( const Int rank, const Int * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Real, rank, dims, &M );
        
        return M;
    }
    
    
    template<> inline MTensor make_MTensor<mcomplex>( const Int rank, const Int * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Complex, rank, dims, &M );
        
        return M;
    }
    
    template<> inline MTensor make_MTensor<Complex>( const Int rank, const Int * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Complex, rank, dims, &M );
        
        return M;
    }
    
    
    template<typename Scal>
    inline MTensor make_MTensor( const std::initializer_list<Int> dims )
    {
        return make_MTensor<Scal>( static_cast<Int>(dims.size()), &dims.begin()[0] );
    }
    
    
    inline void disown( MTensor & M )
    {
        libData->MTensor_disown(M);
    }
    
    
    template<typename Scal>
    class MTensorWrapper
    {
        static_assert( 
            std::is_same_v<Scal,Real> || std::is_same_v<Scal,Int> || std::is_same_v<Scal,Complex>,
            "Only the types Real (double), Int (int64_t), and Complex (std::complex<double>) are allowed."
        );
        
    private: 
        
        MTensor tensor;
        
        Scal * tensor_data;
        
    public:
        
        MTensorWrapper()
        :   tensor      {nullptr}
        ,   tensor_data {nullptr}
        {}
                    
        MTensorWrapper( const MTensor & A )
        :   tensor      { A }
        ,   tensor_data { mma::data<Scal>(tensor) }
        {}
        
        MTensorWrapper( MArgument arg )
        :   tensor      { get<MTensor>(arg) }
        ,   tensor_data { mma::data<Scal>(tensor) }
        {}
        
        MTensorWrapper( const Int rank, const Int * dims )
        :   tensor      { make_MTensor<Scal>( rank, dims ) }
        ,   tensor_data { mma::data<Scal>(tensor) }
        {}
        
        MTensorWrapper( const std::initializer_list<Int> dims )
        :   tensor      { make_MTensor<Scal>(dims) }
        ,   tensor_data { mma::data<Scal>(tensor) }
        {}
        
        MTensorWrapper( const std::initializer_list<Int> dims, const Scal * a )
        :   tensor      { make_MTensor<Scal>(dims) }
        ,   tensor_data { mma::data<Scal>(tensor) }
        {
            //copy_buffer( a, tensor_data, Size() );
            std::copy_n( a, Size(), tensor_data );
        }
        
        Scal * begin()
        {
            return this->data();
        }
        
        const Scal * begin() const
        {
            return this->data();
        }
        
        Scal * end()
        {
            return &tensor_data[Size()];
        }
        
        const Scal * end() const
        {
            return &tensor_data[Size()];
        }
        
        MTensor Tensor() const
        {
            return tensor;
        }

        Int Rank() const 
        {
            return libData->MTensor_getRank(tensor);
        }

        Int Size() const 
        {
            return libData->MTensor_getFlattenedLength(tensor);
        }

        std::size_t size() const
        {
            return static_cast<std::size_t>(libData->MTensor_getFlattenedLength(tensor));
        }

        /// Free the referenced Tensor; same as \c MTensor_free
        /**
         * Tensors created by the library with functions such as \ref makeVector() must be freed
         * after use unless they are returned to _Mathematica_.
         *
         * Warning: multiple \ref TensorRef objects may reference the same \c MTensor.
         * Freeing the \c MTensor invalidates all references to it.
         */
        void Free() const 
        {
            libData->MTensor_free(tensor);
        }

        void Disown() const
        {
            libData->MTensor_disown(tensor);
        }

        void DisownAll() const 
        {
            libData->MTensor_disownAll(tensor);
        }

        mint ShareCount() const {
            return libData->MTensor_shareCount(tensor);
        }

        MTensorWrapper Clone() const
        {
            MTensor C = nullptr;
            
            (void)libData->MTensor_clone(tensor, &C);
            
//            int err = libData->MTensor_clone(tensor, &C);
//            
//            if (err) 
//            {
//                throw LibraryError("MTensor_clone() failed.", err);
//            }
            
            return MTensorWrapper( C );
        }

        
        
        
        const Int * Dims() const
        {
            return libData->MTensor_getDimensions(tensor);
        }
        
        const Int * Dimensions() const
        {
            return Dims();
        }
        
        Int Dim( const Int i ) const
        {
            return libData->MTensor_getDimensions(tensor)[i];
        }
        
        Int Dimension( const Int i ) const
        {
            return Dim(i);
        }

        Scal * data()
        {
            return tensor_data;
        }
        
        const Scal * data() const
        {
            return tensor_data;
        }


        Scal & operator[]( const Int i )
        {
            return tensor_data[i];
        }
        
        const Scal & operator[]( const Int i ) const
        {
            return tensor_data[i];
        }
        
    }; // namespace MTensorWrapper
    
} // namespace mma


extern "C" DLLEXPORT Int WolframLibrary_getVersion()
{
    return WolframLibraryVersion;
}

extern "C" DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData)
{
    mma::libData = libData;
    return LIBRARY_NO_ERROR;
}

extern "C" DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData)
{
    return;
}
