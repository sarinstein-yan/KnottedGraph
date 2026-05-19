#ifdef LTEMPLATE_H
    
    template<bool replace_inftyQ = false, typename S, IntQ Int, Size_T alignment,
        class = typename std::enable_if_t<mma::HasTypeQ<S>>
    >
    inline mma::TensorRef<mma::Type<S>> to_MTensorRef(
       cref<TENSOR_T<S,Int,alignment>> A
    )
    {
        using T = mma::Type<S>;
        
        auto B = mma::makeTensor<T>( A.Rank(), A.Dims() );
        
        if constexpr ( SameQ<T,double> && replace_inftyQ )
        {
            copy_buffer_replace_infty(A.data(),B.data(),A.Size());
        }
        else
        {
            A.Write(B.data());
        }
        
        return B;
    }

    
#endif

#ifdef TENSORS_MMA_HPP

    template<bool replace_inftyQ = false, typename S, IntQ Int, Size_T alignment,
        class = typename std::enable_if_t<mma::HasTypeQ<S>>
    >
    inline mma::MTensorWrapper<mma::Type<S>> to_MTensorWrapper(
        cref<TENSOR_T<S,Int,alignment>> A
    )
    {
        using T = mma::Type<S>;
        
        mma::MTensorWrapper<T> B ( A.Rank(), A.Dims() );
        
        if constexpr ( SameQ<T,double> && replace_inftyQ )
        {
            copy_buffer_replace_infty(A.data(),B.data(),A.Size());
        }
        else
        {
            A.Write(B.data());
        }
        
        return B;
    }

//    template<typename Real, IntQ Int, Size_T alignment,
//        class = typename std::enable_if_t<FloatQ<Real>>
//    >
//    inline mma::MTensorWrapper<std::complex<mreal>> to_MTensorWrapper( cref<TENSOR_T<std::complex<Real>,Int,alignment>> A
//    )
//    {
//        mma::MTensorWrapper<std::complex<mreal>> B ( A.Rank(), A.Dims() );
//        A.Write( B.data() );
//        return B;
//    }
//
//    template<typename J, IntQ Int, Size_T alignment,
//        class = typename std::enable_if_t<IntQ<J>>
//    >
//    inline mma::MTensorWrapper<mint> to_MTensorWrapper(
//        cref<TENSOR_T<J,Int,alignment>> A
//    )
//    {
//        mma::MTensorWrapper<mint> B ( A.Rank(), A.Dims() );
//        A.Write( B.data() );
//        return B;
//    }

#endif
