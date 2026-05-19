#include "Tiny_Matrix_Common.hpp"

//######################################################
//##                     Memory                       ##
//######################################################
    
public:

    static constexpr Int AmbientDimension()
    {
        return n;
    }

    static constexpr Int Dim( const Int i )
    {
        if( i == 0 )
        {
            return n;
        }
        if( i == 1 )
        {
            return n;
        }
        return Int(0);
    }

    static constexpr Int Dimension( const Int i )
    {
        return Dim(i);
    }

    static constexpr Int RowCount()
    {
        return n;
    }

    static constexpr Int ColCount()
    {
        return n;
    }

    TOOLS_FORCE_INLINE void SetZero()
    {
        if constexpr ( n > Int(0) )
        {
            zerofyUpper<0>();
        }
    }

    TOOLS_FORCE_INLINE void ZerofyUpper()
    {
        if constexpr ( n > Int(0) )
        {
            zerofyUpper<0>();
        }
    }

    TOOLS_FORCE_INLINE void ZerofyLower()
    {
        if constexpr ( n > Int(0) )
        {
            zerofyLower<0>();
        }
    }
    
    TOOLS_FORCE_INLINE void Fill( cref<Scal> init )
    {
        FillUpper(init);
    }
    

    TOOLS_FORCE_INLINE void FillUpper( cref<Scal> init )
    {
        if constexpr ( n > Int(0) )
        {
            fillUpper<0>(init);
        }
    }

    TOOLS_FORCE_INLINE void FillLower( cref<Scal> init )
    {
        if constexpr ( n > Int(0) )
        {
            fillLower<0>(init);
        }
    }
    template<typename S>
    TOOLS_FORCE_INLINE void Read( cptr<S> B )
    {
        if constexpr ( n > Int(0) )
        {
            read<0>(B);
        }
    }

    template<typename S>
    TOOLS_FORCE_INLINE void Read( cptr<S> B, const Int ldB )
    {
        if constexpr ( n > Int(0) )
        {
            read<0>(B,ldB);
        }
    }
    
    template<typename S>
    TOOLS_FORCE_INLINE void Write( mptr<S> B ) const
    {
        if constexpr ( n > Int(0) )
        {
            write<0>(B);
        }
    }

    template<typename S>
    TOOLS_FORCE_INLINE void Write( mptr<S> B, const Int ldB ) const
    {
        if constexpr ( n > Int(0) )
        {
            write<0>(B,ldB);
        }
    }
    
protected:
    
    // Trying to use compile-time loops to unroll these operations.
    
    template<Int k>
    TOOLS_FORCE_INLINE void zerofyUpper()
    {
        zerofy_buffer<n-k>( &A[k][k] );
        
        if constexpr ( k + Int(1) < n )
        {
            zerofyUpper<k+1>();
        }
    }

    template<Int k>
    TOOLS_FORCE_INLINE void zerofyLower()
    {
        zerofy_buffer<k>( &A[k][0] );
        
        if constexpr ( k + Int(1) < n )
        {
            zerofyLower<k+1>();
        }
    }
    
    template<Int k>
    TOOLS_FORCE_INLINE void fillUpper( const Scal init )
    {
        fill_buffer<n-k>( &A[k][k], init );
        
        if constexpr ( k + Int(1) < n )
        {
            fillUpper<k+1>(init);
        }
    }

    template<Int k>
    TOOLS_FORCE_INLINE void fillLower( const Scal init )
    {
        fill_buffer<k>( &A[k][0], init );
        
        if constexpr ( k + Int(1) < n )
        {
            fillLower<k+1>(init);
        }
    }
    
    template<Int k, typename S>
    TOOLS_FORCE_INLINE void read( cptr<S> const B )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k + Int(1) )
        {
            read<k+1>( &B[n+1] );
        }
    }

    template<Int k, typename S>
    TOOLS_FORCE_INLINE void read( cptr<S> B, const Int ldB )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k + Int(1) )
        {
            read<k+1>( &B[ldB+1], ldB );
        }
    }
    
    template<Int k, typename S>
    TOOLS_FORCE_INLINE void write( mptr<S> B ) const
    {
        copy_buffer<n-k>( &A[k][k], B );
        
        if constexpr ( n > k + Int(1) )
        {
            write<k+1>( &B[n+1] );
        }
    }

    template<Int k, typename S>
    TOOLS_FORCE_INLINE void write( mptr<S> B, const Int ldB ) const
    {
        copy_buffer<n-k>( &A[k][k], B );
        
        if constexpr ( n > k + Int(1) )
        {
            write<k+1>( &B[ldB+1], ldB );
        }
    }



//######################################################
//##                  Arithmetic                      ##
//######################################################
       
public:

    friend Class_T operator+( cref<Class_T> x, cref<Class_T> y )
    {
        Class_T z;
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                z.A[i][j] = x.A[i][j] + y.A[i][j];
            }
        }
        return z;
    }
    
    TOOLS_FORCE_INLINE void Conjugate( mref<Class_T> B ) const
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                B.A[i][j] = Conj(A[i][j]);
            }
        }
    }
    
    
    template<class T>
    TOOLS_FORCE_INLINE mref<Class_T> operator+=( cref<T> lambda )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] += lambda;
            }
        }
        
        return *this;
    }

    template<class T>
    TOOLS_FORCE_INLINE mref<Class_T> operator-=( cref<T> lambda )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] -= lambda;
            }
        }
        
        return *this;
    }
    
    template<class T>
    TOOLS_FORCE_INLINE mref<Class_T> operator*=( cref<T> lambda )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] *= lambda;
            }
        }
        
        return *this;
    }


    TOOLS_FORCE_INLINE Real MaxNorm() const
    {
        Real max = 0;
        
        if constexpr ( Scalar::RealQ<Scal> )
        {
            for( Int i = 0; i < n; ++i )
            {
                for( Int j = i; j < n; ++j )
                {
                    max = Tools::Max( max, Abs(A[i][j]) );
                }
            }
            
            return max;
        }
        else
        {
            for( Int i = 0; i < n; ++i )
            {
                for( Int j = i; j < n; ++j )
                {
                    max = Tools::Max( max, AbsSquared(A[i][j]) );
                }
            }
            
            return Sqrt(max);
        }   
    }

    [[nodiscard]] friend std::string ToString( cref<Class_T> M )
    {
        return OutString::FromMatrix(
            [&M]( const Int i, const Int j ) { return M.A[i][j]; }, n, n
        );
    }

    inline friend std::ostream & operator<<( std::ostream & s, cref<Class_T> M )
    {
        return s << OutString::FromMatrix(
            [&M]( const Int i, const Int j ) { return M.A[i][j]; }, n, n
        );
    }
