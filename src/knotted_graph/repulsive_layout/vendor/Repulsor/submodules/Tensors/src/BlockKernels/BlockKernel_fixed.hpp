#pragma once

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int NRHS_, bool fixed,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        IntQ Int_, IntQ LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        bool x_RM, bool x_intRM, bool x_copy, bool x_prefetch,
        bool y_RM, bool y_intRM,
        bool use_fma
    >
    class alignas( ObjectAlignment ) BlockKernel_fixed
    {
        static_assert(ArithmeticQ<Scal_>,"");
        static_assert(ArithmeticQ<Scal_in_>,"");
        static_assert(ArithmeticQ<Scal_out_>,"");
        
    public:
        
        using Scal     = Scal_;
        using Scal_in  = Scal_in_;
        using Scal_out = Scal_out_;
        using Int      = Int_;
        using LInt     = LInt_;
        
        static constexpr Int NRHS = NRHS_;
        static constexpr Int MAX_NRHS = NRHS_;
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        static constexpr Int ROWS_SIZE = ROWS_ * NRHS_;
        static constexpr Int COLS_SIZE = COLS_ * NRHS_;
        
    protected:
        
        mptr<Scal>     A       = nullptr;
        cptr<Scal>     A_const = nullptr;
        const Scal_out alpha   = 0;
        cptr<Scal_in>  X       = nullptr;
        const Scal_out beta    = 0;
        mptr<Scal_out> Y       = nullptr;

        
        const Scal_in  * restrict x_from = nullptr;
//              Scal_out * restrict y_to   = nullptr;
        
        
        Tiny::Matrix<x_intRM ? COLS : NRHS, x_intRM ? NRHS : COLS, Scal,Int> x;
        Tiny::Matrix<y_intRM ? ROWS : NRHS, y_intRM ? NRHS : ROWS, Scal,Int> y;

        const Int nrhs = 1;
        const Int rows_size = ROWS;
        const Int cols_size = COLS;
        
    public:
        
        explicit BlockKernel_fixed( mptr<Scal> A_ )
        :   A       ( A_      )
        ,   A_const ( nullptr )
        ,   alpha   ( 0       )
        ,   X       ( nullptr )
        ,   beta    ( 0       )
        ,   Y       ( nullptr )
        {}

        BlockKernel_fixed(
            cptr<Scal> A_,
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            Int nrhs_
        )
        :   A         ( nullptr          )
        ,   A_const   ( A_               )
        ,   alpha     ( alpha_           )
        ,   X         ( X_               )
        ,   beta      ( beta_            )
        ,   Y         ( Y_               )
        ,   nrhs      ( nrhs_       )
        ,   rows_size ( ROWS * nrhs )
        ,   cols_size ( COLS * nrhs )
        {
            if( fixed && (NRHS != nrhs) )
            {
                eprint(ClassName()+"nrhs != NRHS");
            }
            
            if( nrhs > NRHS)
            {
                eprint(ClassName()+"nrhs > NRHS");
            }
        }
        
        // Copy constructor
        BlockKernel_fixed( const BlockKernel_fixed & other )
        :   A         ( other.A           )
        ,   A_const   ( other.A_const     )
        ,   alpha     ( other.alpha       )
        ,   X         ( other.X           )
        ,   beta      ( other.beta        )
        ,   Y         ( other.Y           )
        ,   nrhs ( other.nrhs_  )
        ,   rows_size ( other.rows_size   )
        ,   cols_size ( other.cols_size   )
        {}
        
        // No default constructor
        BlockKernel_fixed() = delete;
        // Destructor
        virtual ~BlockKernel_fixed() = default;
        // Copy constructor
        BlockKernel_fixed( const BlockKernel_fixed & other ) = default;
        // Copy assignment operator
        BlockKernel_fixed & operator=( const BlockKernel_fixed & other ) = default;
        // Move constructor
        BlockKernel_fixed( BlockKernel_fixed && other ) = default;
        // Move assignment operator
        BlockKernel_fixed & operator=( BlockKernel_fixed && other ) = default;


    public:
        
        static constexpr Int RowCount()
        {
            return ROWS;
        }
        
        static constexpr Int ColCount()
        {
            return COLS;
        }
        
        Int RightHandSideCount() const
        {
            return RhsCount();
        }
        
//        LInt NonzeroCount() const = 0;
        
//        void TransposeBlock( const LInt from, const LInt to ) const = 0;
        
        
        
        TOOLS_FORCE_INLINE Int ColsSize() const
        {
            if constexpr ( fixed )
            {
                return COLS_SIZE;
            }
            else
            {
                return cols_size;
            }
        }
    
        TOOLS_FORCE_INLINE Int RowsSize() const
        {
            if constexpr ( fixed )
            {
                return ROWS_SIZE;
            }
            else
            {
                return rows_size;
            }
        }

        TOOLS_FORCE_INLINE Int RhsCount() const
        {
            if constexpr ( fixed )
            {
                return NRHS;
            }
            else
            {
                return nrhs;
            }
        }
        
        TOOLS_FORCE_INLINE void FMA( cref<Scal> a, cref<Scal> b, mref<Scal> c ) const
        {
            if constexpr ( use_fma )
            {
                c = std::fma(a,b,c);
            }
            else
            {
                c += a * b;
            }
        }
        
        TOOLS_FORCE_INLINE void ReadX( const Int j_global )
        {
            if constexpr ( x_copy )
            {
                x_from = &X[ColsSize() * j_global];
                
                if constexpr ( x_RM )
                {
                    if constexpr ( x_intRM )
                    {
                        if constexpr ( fixed )
                        {
                            copy_buffer( x_from, &x[0][0], ColsSize() );
                        }
                        else
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                                {
                                    x[j][k] = static_cast<Scal>( x_from[(fixed ? NRHS : nrhs)*j+k] );
                                }
                            }
                        }
                    }
                    else
                    {
                        TOOLS_LOOP_UNROLL_FULL
                        for( Int j = 0; j < COLS; ++j )
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                x[k][j] = static_cast<Scal>( x_from[(fixed ? NRHS : nrhs)*j+k] );
                            }
                        }
                    }
                }
                else
                {
                    if constexpr ( x_intRM )
                    {
                        TOOLS_LOOP_UNROLL_FULL
                        for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                x[j][k] = static_cast<Scal>( x_from[COLS*k+j] );
                            }
                        }
                    }
                    else
                    {
                        // Here we are allowed to copy the full slice because internal x is column major.
                        copy_buffer( x_from, &x[0][0], ColsSize() );
                    }
                }
            }
            else
            {
                x_from = &X[ColsSize() * j_global];
            }
        }
        
        TOOLS_FORCE_INLINE void Prefetch( const LInt k_global, const Int j_next )
        {
            (void)k_global;
            
            if constexpr ( x_prefetch )
            {
                // X is accessed in an unpredictable way; let's help with a prefetch statement.
                if constexpr ( fixed )
                {
                    prefetch_buffer<COLS_SIZE,0,0>( &X[COLS_SIZE * j_next] );
                }
                else
                {
                    prefetch_buffer<0,0>( &X[cols_size * j_next], cols_size );
                }
            }
            else
            {
                (void)j_next;
            }
            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
        }
        
        TOOLS_FORCE_INLINE Scal_out get_cast_y( const Int i, const Int k ) const
        {
            if constexpr ( y_intRM )
            {
                return static_cast<Scal_out>(y[i][k]);
            }
            else
            {
                return static_cast<Scal_out>(y[k][i]);
            }
        }
        
        TOOLS_FORCE_INLINE mref<Scal> get_y( const Int i, const Int k )
        {
            if constexpr ( y_intRM )
            {
                return y[i][k];
            }
            else
            {
                return y[k][i];
            }
        }
        
//        TOOLS_FORCE_INLINE Scal get_cast_x( const Int j, const Int k )
//        {
//            if constexpr ( x_intRM )
//            {
//                return static_cast<Scal>( x_from[(fixed ? NRHS : nrhs)*j+k];
//            }
//            else
//            {
//                return static_cast<Scal>( x_from[COLS*k+j];
//            }
//        }
        
        TOOLS_FORCE_INLINE Scal get_x( const Int j, const Int k )
        {
            if constexpr ( x_copy )
            {
                if constexpr ( x_intRM )
                {
                    return x[j][k];
                }
                else
                {
                    return x[k][j];
                }
            }
            else
            {
                if constexpr ( x_RM )
                {
                    return static_cast<Scal>(x_from[(fixed ? NRHS : nrhs)*j+k]);
                }
                else
                {
                    return static_cast<Scal>(x_from[COLS*k+j]);
                }
            }
        }
        
        
        TOOLS_FORCE_INLINE void CleanseY()
        {
            // Clear the local vector chunk of the kernel.
//            zerofy_buffer<ROWS_SIZE>( &y[0][0] );
            y.SetZero();
        }
        
        TOOLS_FORCE_INLINE void WriteY( const Int i_global ) const
        {
            TOOLS_MAKE_FP_FAST()
            
            mptr<Scal_out> y_to = &Y[ RowsSize() * i_global];
            
            if constexpr ( alpha_flag == Scalar::Flag::Plus )
            {
                // alpha == 1;
                if constexpr ( beta_flag == Scalar::Flag::Zero )
                {
                    if constexpr (y_RM)
                    {
                        if constexpr ( y_intRM)
                        {
                            if constexpr ( fixed )
                            {
//                                copy_buffer<ROWS_SIZE>( &y[0][0], y_to );
                                y.Write(y_to);
                            }
                            else
                            {
                                copy_buffer( &y[0][0], y_to, RowsSize() );
                            }
                        }
                        else
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                                {
                                    y_to[(fixed ? NRHS : nrhs)*i+k] = get_cast_y(i,k);
                                }
                            }
                        }
                    }
                    else
                    {
                        // y is not row-major
                        
                        if ( y_intRM )
                        {
                            //transpose
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    y_to[ROWS*k+i] = get_cast_y(i,k);
                                }
                            }
                        }
                        else
                        {
                            if constexpr ( fixed )
                            {
//                                copy_buffer<NRHS>( &y[0][0], y_to );
                                y.Write( y_to );
                            }
                            else
                            {
                                copy_buffer( &y[0][0], y_to, nrhs );
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == Scalar::Flag::Plus )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                y_to[(fixed ? NRHS : nrhs)*i+k] += get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] += get_cast_y(i,k);
                            }
                        }
                    }
                }
                else
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                y_to[(fixed ? NRHS : nrhs)*i+k] = get_cast_y(i,k) + beta * y_to[(fixed ? NRHS : nrhs)*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = get_cast_y(i,k) + beta * y_to[ROWS*k+i];
                            }
                        }
                    }
                }
            }
            else if constexpr ( alpha_flag == Scalar::Flag::Zero )
            {
                if constexpr ( beta_flag == Scalar::Flag::Zero )
                {
                    if constexpr ( fixed )
                    {
                        zerofy_buffer<ROWS_SIZE>( y_to );
                    }
                    else
                    {
                        zerofy_buffer( y_to, RowsSize() );
                    }
                }
                else if constexpr ( beta_flag == Scalar::Flag::Plus )
                {
                    // do nothing;
                }
                else
                {
                    if constexpr ( fixed )
                    {
                        scale_buffer<ROWS_SIZE>( beta, y_to );
                    }
                    else
                    {
                        scale_buffer( beta, y_to, RowsSize() );
                    }
                }
            }
            else // alpha_flag == Scalar::Flag::Generic or Scalar::Flag::Minus
            {
                // general alpha
                
                if constexpr ( beta_flag == Scalar::Flag::Zero )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                y_to[(fixed ? NRHS : nrhs)*i+k] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == Scalar::Flag::Plus )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                y_to[(fixed ? NRHS : nrhs)*i+k] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                }
                else // // beta_flag == Scalar::Flag::Generic or Scalar::Flag::Minus
                {
                    // general alpha and general beta
                    
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                y_to[(fixed ? NRHS : nrhs)*i+k] = alpha * get_cast_y(i,k) + beta * y_to[(fixed ? NRHS : nrhs)*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = alpha * get_cast_y(i,k) + beta * y_to[ROWS*k+i];
                            }
                        }
                    }
                }
            }
        }
        
        TOOLS_FORCE_INLINE void WriteYZero( const Int i_global ) const
        {
            // CAUTION! We cannot use i_global here because BeginRow() has not been in an empty row!
            mptr<Scal_out> y_to = &Y[ RowsSize() * i_global ];
            
            if constexpr ( beta_flag == Scalar::Flag::Zero )
            {
                if constexpr ( fixed )
                {
                    zerofy_buffer<ROWS_SIZE>( y_to );
                }
                else
                {
                    zerofy_buffer( y_to, RowsSize() );
                }
            }
            else if constexpr ( beta_flag == Scalar::Flag::Plus )
            {
                // do nothing;
            }
            else
            {
                if constexpr ( fixed )
                {
                    scale_buffer<ROWS_SIZE>(beta, y_to);
                }
                else
                {
                    scale_buffer( beta, y_to, RowsSize() );
                }
            }
        }

        
    public:
        
        std::string ClassName() const
        {
            return std::string("BlockKernel_fixed")+"<"
                +ToString(ROWS)+","+ToString(COLS)+","+ToString(NRHS)+","+ToString(fixed)
            
            +","+TypeName<Scal>+","+TypeName<Scal_in>+","+TypeName<Scal_out>
            +","+TypeName<Int>+","+TypeName<LInt>
            +","+ToString(alpha_flag)+","+ToString(beta_flag)

            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)+
            +">";
        }
  
    };

} // namespace Tensors

