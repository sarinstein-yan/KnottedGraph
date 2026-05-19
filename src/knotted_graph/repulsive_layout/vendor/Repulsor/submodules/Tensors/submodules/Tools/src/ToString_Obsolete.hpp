#pragma once

namespace Tools
{
    template<typename Scal, IntQ Int, IntQ Int2, IntQ Int3, NonIntQ F>
    [[nodiscard]] std::string ArrayToString(
        cptr<Scal> a,
        cptr<Int>  const dims,
        cptr<Int2> const lds,
        Int3 rank,
        F && fun,
        std::string line_prefix = ""
    )
    {
        std::string s;
        
        // TODO: Find a good reserve strategy.
//        s.reserve( ToSize_T(dims[0]) * ( Size_T(4) + line_prefix.size() + Size_T(6) * ToSize_T(dims[1]) ) );
        
        Size_T r = ToSize_T(rank);
        
        if( r <= Size_T(0) )
        {
            s += fun(a[0]);
        }
        else if( r == Size_T(1) )
        {
            s += line_prefix;
            s += "{ ";
            
            if( dims[0] > Int(0) )
            {
                s += fun(a[0]);
            }
            
            for( Int i = 1; i < dims[0]; ++i )
            {
                s += ", ";
                s += fun(a[i]);
            }
            
            s += " }";
        }
        else
        {
            std::string new_line_prefix = line_prefix + "\t";
            
            s += line_prefix;
            s += "{\n";
            
            if( dims[0] > Int(0) )
            {
                s += ArrayToString( a, &dims[1], &lds[1], r - Size_T(1), fun, new_line_prefix );
            }
            
            for( Int i = 1; i < dims[0]; ++i )
            {
                s += ",\n";
                s += ArrayToString( &a[lds[0]*i], &dims[1], &lds[1], r - Size_T(1), fun, new_line_prefix );
            }
            
            s += "\n";
            
            s += line_prefix;
            s += "}";
        }
        
        return s;
    }
    
    template<typename Scal, IntQ Int, IntQ Int2, IntQ Int3>
    [[nodiscard]] std::string ArrayToString(
        cptr<Scal> a,
        cptr<Int>  dims,
        cptr<Int2> lds,
        Int3 rank,
        std::string line_prefix  = std::string("")
    )
    {
        return ArrayToString(
            a,dims,lds,rank,[]( cref<Scal> a ){ return ToString(a); },line_prefix
        );
    }
    
    template<typename Scal, IntQ Int, IntQ Int2, NonIntQ F>
    [[nodiscard]] std::string ArrayToString(
        cptr<Scal> a,
        cptr<Int> dims,
        Int2 rank,
        F && fun,
        std::string line_prefix = ""
    )
    {
        const Size_T r = ToSize_T(rank);
        
        if( r >= Size_T(0) )
        {
            std::vector<Int> lds (r);

            if( r >= Size_T(1) )
            {
                lds[r-1] = Size_T(1);
//
//                    for( Size_T i = r - Size_T(1); i --> Size_T(0);  )
//                    {
//                        lds[i] = lds[i+ Size_T(1)] * dims[i+ Size_T(1)];
//                    }
                
                
                for( Size_T i = r; i --> Size_T(1);  )
                {
                    lds[i - Size_T(1)] = lds[i] * dims[i];
                }
            }
            
            return ArrayToString( a, dims, lds.data(), rank, fun, line_prefix );
        }
        else
        {
            return std::string();
        }
    }
    template<typename Scal, IntQ Int, IntQ Int2>
    [[nodiscard]] std::string ArrayToString(
        cptr<Scal> a,
        cptr<Int> dims,
        Int2 rank,
        std::string line_prefix = std::string("")
    )
    {
        return ArrayToString(
            a,dims, rank,
            []( cref<Scal> a ){ return ToString(a); },
            line_prefix
        );
    }
    
    template<typename Scal, IntQ Int, NonIntQ F>
    [[nodiscard]] std::string ArrayToString(
        cptr<Scal> a,
        std::initializer_list<Int> dims,
        F && fun,
        std::string line_prefix = ""
    )
    {
        return ArrayToString( a, &*dims.begin(), dims.size(), fun, line_prefix );
    }

    template<typename Scal, IntQ Int>
    [[nodiscard]] std::string ArrayToString(
        cptr<Scal> a,
        std::initializer_list<Int> dims,
        std::string line_prefix = ""
    )
    {
        return ArrayToString(
            a, &*dims.begin(), dims.size(),
            []( cref<Scal> a ){ return ToString(a); },
            line_prefix
        );
    }

    
//    template<typename T, NonIntQ F>
//    [[nodiscard]] std::string ToString(
//        cref<std::vector<T>> v,
//        F && fun,
//        std::string line_prefix
//    )
//    {
//        const Size_T dim = v.size();
//        return ArrayToString( &v[0], &dim, Size_T(1), fun, line_prefix );
//    }
//    
//    template<typename T>
//    [[nodiscard]] std::string ToString(
//        const std::vector<T> & v,
//        std::string line_prefix = ""
//    )
//    {
//        return ToString( v, []( cref<T> a ){ return ToString(a); }, line_prefix );
//    }
//    
//    template<typename T, Size_T N, NonIntQ F>
//    [[nodiscard]] std::string ToString(
//        const std::array<T,N> & v,
//        F && fun,
//        std::string line_prefix = ""
//    )
//    {
//        const Size_T dim = N;
//        return ArrayToString( &v[0], &dim, Size_T(1), fun, line_prefix );
//    }
//    
//    template<typename T, Size_T N>
//    [[nodiscard]] std::string ToString(
//        const std::array<T,N> & v,
//        std::string line_prefix = ""
//    )
//    {
//        return ToString( v, []( cref<T> a ){ return ToString(a); }, line_prefix );
//    }

    template<Size_T N_ = VarSize, typename T, NonIntQ F, IntQ Int = Size_T>
    [[nodiscard]] std::string VectorString(
        cptr<T> A,
        cref<std::string> prefix,
        cref<std::string> infix,
        cref<std::string> suffix,
        const Int n,
        F && fun
    )
    {
        constexpr Int N = static_cast<Int>(N_);
        
        std::string s;
        // TODO: Find a good reserve strategy.
//        s.reserve( ToSize_T(n) * (infix.size() + Size_T(4) ) + prefix.size() + suffix.size() );
        
        s += prefix;
        
        if( ((N_ > VarSize) ? N : n) > Int(0) )
        {
            s += fun(A[0]);
        }
        
        for( Int i = 1; i < ((N_ > VarSize) ? N : n); ++i )
        {
            s += infix;
            s += fun(A[i]);
        }
        
        s  += suffix;
        
        return s;
    }
    
    
    template<Size_T N_ = VarSize, typename T, IntQ Int = Size_T>
    [[nodiscard]] std::string VectorString(
        cptr<T> A,
        cref<std::string> prefix,
        cref<std::string> infix,
        cref<std::string> suffix,
        const Int n = static_cast<Int>(N_)
    )
    {
        return VectorString<N_>(
            A, prefix, infix, suffix, n, []( cref<T> a ){ return ToString(a); }
        );
    }
    
    template<Size_T M_ = VarSize, Size_T N_ = VarSize, typename T, NonIntQ F, IntQ Int = Size_T>
    [[nodiscard]] std::string MatrixString(
        cptr<T> A, const Size_T ldA,
        cref<std::string> header,
        cref<std::string> row_prefix,
        cref<std::string> row_infix,
        cref<std::string> row_suffix,
        cref<std::string> row_sep,
        cref<std::string> footer,
        const Int m,
        const Int n,
        F && fun
    )
    {
        constexpr Int M = static_cast<Int>(M_);
        constexpr Int N = static_cast<Int>(N_);
        
        std::string s;
        
        s += header;
        
        if( ((M_ > VarSize) ? M : m) > Int(0) )
        {
            s += VectorString<N>(&A[0],row_prefix,row_infix,row_suffix,n,fun);
        }
        
        for( Int i = 1; i < ((M_ > VarSize) ? M : m); ++i )
        {
            s += row_sep;
            
            s += VectorString<N>(&A[ldA * i],row_prefix,row_infix,row_suffix,n,fun);
        }

        s += footer;
        
        return s;
    }
    
    template<Size_T M_ = VarSize, Size_T N_ = VarSize, typename T, IntQ Int = Size_T>
    [[nodiscard]] std::string MatrixString(
        cptr<T> A, const Size_T ldA,
        cref<std::string> header,
        cref<std::string> row_prefix,
        cref<std::string> row_infix,
        cref<std::string> row_suffix,
        cref<std::string> row_sep,
        cref<std::string> footer,
        const Int m = static_cast<Int>(M_),
        const Int n = static_cast<Int>(N_)
    )
    {
        return MatrixString<M_,N_>(
            A, ldA, header, row_prefix, row_infix, row_suffix, row_sep, footer, m, n,
            []( cref<T> a ){ return ToString(a); }
        );
    }
    
    template<typename T>
    [[nodiscard]] std::string MatrixStringTSV(
        const Size_T m, const Size_T n, cptr<T> A, const Size_T ldA
    )
    {
        return MatrixString(
            A, ldA, "", "", "\t", "", "\n", "", m, n,
            []( cref<T> a ){ return ToString(a); }
        );
    }
    
    template<typename T>
    [[nodiscard]] std::string MatrixStringCSV(
        const Size_T m, const Size_T n, cptr<T> A, const Size_T ldA
    )
    {
        return MatrixString(
            A, ldA, "", "", ",", "", "\n", "", m, n,
            []( cref<T> a ){ return ToString(a); }
        );
    }
    
} // namespace Tools

