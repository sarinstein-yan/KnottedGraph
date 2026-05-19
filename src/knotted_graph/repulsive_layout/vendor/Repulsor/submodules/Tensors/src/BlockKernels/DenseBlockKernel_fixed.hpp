#pragma once

#define CLASS DenseBlockKernel_fixed

#define BASE  BlockKernel_fixed<                            \
    ROWS_,COLS_,NRHS_,fixed,                                \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_,LInt_,                                             \
    alpha_flag, beta_flag,                                  \
    x_RM, x_intRM, x_copy, x_prefetch,                      \
    y_RM, y_intRM,                                          \
    useFMA                                                  \
>

//template<
//    int ROWS_, int COLS_, int NRHS_, bool fixed,
//    typename Scal_, typename Scal_in_, typename Scal_out_,
//    IntQ Int_, IntQ LInt_,
//    int alpha_flag, int beta_flag,
//    bool a_RM  = true, bool a_intRM = false,  bool a_copy = true,
//    bool x_RM  = true, bool x_intRM = false,  bool x_copy = true,  bool x_prefetch = true,
//    bool y_RM  = true, bool y_intRM = false,
//    int method = 1, int loop   = 2,
//    bool useFMA = false
//>

// TODO: This one needs a great overhaul in terms of Tiny::Matrix routines.
namespace Tensors
{
    // I picked the default values from benchmarks for
    // ROWS_ = 4, COLS_ = 4, NRHS_ = 3, alpha_flag = 1, beta_flag = 0, and doubles for all floating point types.
    template<
        int ROWS_, int COLS_, int NRHS_, bool fixed,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        IntQ Int_, IntQ LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        bool a_RM,      bool a_intRM,   bool a_copy,
        bool x_RM,      bool x_intRM,   bool x_copy,    bool x_prefetch,
        bool y_RM,      bool y_intRM,
        int method,     int loop,
        bool useFMA
    >
    class CLASS : public BASE
    {
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;

        using Int        = Int_;
        using LInt       = LInt_;
        
        using BASE::ROWS;
        using BASE::COLS;
        using BASE::NRHS;
        
        using BASE::RowsSize;
        using BASE::ColsSize;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::RhsCount;
        
        static constexpr LInt BLOCK_NNZ = ROWS * COLS;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::y;
        
        using BASE::ReadX;
        using BASE::get_x;
        using BASE::get_y;
        using BASE::nrhs;
        
        const Scal * restrict a_from = nullptr;
        
//        alignas(ALIGNMENT) Scal a [(a_intRM)?ROWS:COLS][(a_intRM)?COLS:ROWS];
        
        Tiny::Matrix<(a_intRM)?ROWS:COLS,(a_intRM)?COLS:ROWS,Scal,Int> a;
        
    public:
        
        explicit CLASS( mptr<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            cptr<Scal>     A_,
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            const Int      nrhs_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_, nrhs_ )
        {}
        
        // No default constructor
        CLASS() = delete;
        // Destructor
        virtual ~CLASS() override = default;
        // Copy constructor
        CLASS( const CLASS & other ) = default;
        // Copy assignment operator
        CLASS & operator=( const CLASS & other ) = default;
        // Move constructor
        CLASS( CLASS && other ) = default;
        // Move assignment operator
        CLASS & operator=( CLASS && other ) = default;
        
    public:
        
        LInt NonzeroCount() const
        {
            return BLOCK_NNZ;
        }
                
        TOOLS_FORCE_INLINE void TransposeBlock( const LInt from, const LInt to ) const
        {
            cptr<Scal> a_from_ = &A[ BLOCK_NNZ * from];
            mptr<Scal> a_to_   = &A[ BLOCK_NNZ * to  ];
            
            if constexpr ( a_RM )
            {
                TOOLS_LOOP_UNROLL_FULL
                for( Int j = 0; j < COLS; ++j )
                {
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        a_to_[ROWS * j + i ] = a_from_[COLS * i + j ];
                    }
                }
            }
            else
            {
                TOOLS_LOOP_UNROLL_FULL
                for( Int i = 0; i < ROWS; ++i )
                {
                    TOOLS_LOOP_UNROLL_FULL
                    for( Int j = 0; j < COLS; ++j )
                    {
                        a_to_[COLS * i + j] = a_from_[ROWS * j + i];
                    }
                }
            }
        }
        
        TOOLS_FORCE_INLINE void ReadA( const LInt k_global )
        {
            // Read matrix.
            if constexpr ( a_copy )
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
                
                if constexpr ( a_RM == a_intRM )
                {
                    a.template Read<Op::Id>( a_from );
                }
                else
                {
                    a.template Read<Op::Trans>( a_from );
                }
            }
            else
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
            }
        }
        
        TOOLS_FORCE_INLINE Scal get_a( const Int i, const Int j ) const
        {
            if constexpr ( a_copy )
            {
                if constexpr ( a_intRM )
                {
                    return a[i][j];
                }
                else
                {
                    return a[j][i];
                }
            }
            else
            {
                if constexpr ( a_RM )
                {
                    return a_from[COLS*i+j];
                }
                else
                {
                    return a_from[ROWS*j+i];
                }
            }
        }
        
        TOOLS_FORCE_INLINE void ApplyBlock( const LInt k_global, const Int j_global )
        {
            TOOLS_MAKE_FP_FAST()
            
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            ReadX( j_global );
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            
            ReadA( k_global );
            
            switch( method )
            {
                case 0:
                {
                    // Do nothing.
                    break;
                }
                case 1:
                {
                    switch ( loop )
                    {
                        case 0:
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    TOOLS_LOOP_UNROLL_FULL
                                    for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 1:
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                                {
                                    TOOLS_LOOP_UNROLL_FULL
                                    for( Int j = 0; j < COLS; ++j )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 2:
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    TOOLS_LOOP_UNROLL_FULL
                                    for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 3:
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                                {
                                    TOOLS_LOOP_UNROLL_FULL
                                    for( Int i = 0; i < ROWS; ++i )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 4:
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    TOOLS_LOOP_UNROLL_FULL
                                    for( Int j = 0; j < COLS; ++j )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 5:
                        {
                            TOOLS_LOOP_UNROLL_FULL
                            for( Int k = 0; k < (fixed ? NRHS : nrhs); ++k )
                            {
                                TOOLS_LOOP_UNROLL_FULL
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    TOOLS_LOOP_UNROLL_FULL
                                    for( Int i = 0; i < ROWS; ++i )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
//                case 2:
//                {
//                    BLAS::gemm<Layout::RowMajor,a_intRM ? Op::Trans : Op::Id,Op::Id>(
//                        a_intRM ? ROWS : COLS, NRHS, a_intRM ? COLS : ROWS,
//                        Scalar::One<Scal>, &a[0][0], a_intRM ? COLS : ROWS,
//                                           &x[0][0], COLS,
//                        Scalar::One<Scal>, &y[0][0], ROWS,
//                    );
//
//                }
                default:
                {}
            }

        }
        
    public:
        
        std::string ClassName() const
        {
            return TOOLS_TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(NRHS)
            +","+ToString(fixed)
            +","+TypeName<Scal>
            +","+TypeName<Scal_in>
            +","+TypeName<Scal_out>
            +","+TypeName<Int>
            +","+TypeName<LInt>
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +","+ToString(a_RM)+","+ToString(a_intRM)+","+ToString(a_copy)
            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)
            +","+ToString(method)
            +","+ToString(loop)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS

