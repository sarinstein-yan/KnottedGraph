#pragma once

#define CLASS DenseBlockKernel_Tiny

#define BASE  BlockKernel_Tiny<                             \
    ROWS_,COLS_,NRHS_,                                      \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_,LInt_,                                             \
    alpha_flag, beta_flag,                                  \
    x_RM, x_prefetch,                                       \
    y_RM                                                    \
>

//template<
//    int ROWS_, int COLS_, int NRHS_,
//    typename Scal_, typename Scal_in_, typename Scal_out_,
//    IntQ Int_, IntQ LInt_,
//    int alpha_flag, int beta_flag,
//    bool a_RM  = true, bool a_intRM = false,  bool a_copy = true,
//    bool x_RM  = true, bool x_prefetch = true,
//    bool y_RM  = true
//>

namespace Tensors
{
    /// I picked the default values from benchmarks for
    /// ROWS_ = 4, COLS_ = 4, NRHS_ = 3, alpha_flag = 1, beta_flag = 0, and doubles for all floating point types.
    
    /// Note: We use the template parameters a_intRM and a_copy only for backwards 
    /// compatibility! The implementation uses internally row-major and copies the
    /// a-block in any case.
    template<
        int ROWS_, int COLS_, int NRHS_,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        IntQ Int_, IntQ LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        bool a_RM, bool a_intRM, bool a_copy,
        bool x_RM, bool x_prefetch,
        bool y_RM
    >
    class CLASS : public BASE
    {
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;

        using Int      = Int_;
        using LInt     = LInt_;
        
        using BASE::ROWS;
        using BASE::COLS;
        using BASE::NRHS;
        
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        
        static constexpr LInt BLOCK_NNZ = ROWS * COLS;
        
        static constexpr Op opA = a_RM ? Op::Id : Op::Trans;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::y;
        
        using BASE::ReadX;
        
//        const Scal * restrict a_from = nullptr;
        
//        Tiny::Matrix<(a_intRM)?ROWS:COLS,(a_intRM)?COLS:ROWS,Scal,Int> a;
        
        Tiny::Matrix<ROWS,COLS,Scal,Int> a;
        
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
            TOOLS_MAKE_FP_FAST()
            
            cptr<Scal> a_from = &A[ BLOCK_NNZ * from];
            mptr<Scal> a_to   = &A[ BLOCK_NNZ * to  ];
            
            
//            if constexpr ( a_RM )
//            {
//                a.template Read <Op::Id>   ( a_from );
//                a.template Write<Op::Trans>( a_to   );
//            }
//            else
//            {
//                a.template Read <Op::Trans>( a_from );
//                a.template Write<Op::Id>   ( a_to   );
//            }
            
            if constexpr ( a_RM )
            {
                for( Int j = 0; j < COLS; ++j )
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        a_to[ROWS * j + i ] = a_from[COLS * i + j ];
                    }
                }
            }
            else
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
                        a_to[COLS * i + j] = a_from[ROWS * j + i];
                    }
                }
            }
        }
        
        TOOLS_FORCE_INLINE void ReadA( const LInt k_global )
        {
            cptr<Scal> a_from = &A_const[BLOCK_NNZ * k_global];
            
            if constexpr ( a_RM )
            {
                a.template Read<Op::Id>( a_from );
            }
            else
            {
                a.template Read<Op::Trans>( a_from );
            }
        }
        
        TOOLS_FORCE_INLINE void ApplyBlock( const LInt k_global, const Int j_global )
        {
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            ReadX( j_global );
            
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            ReadA( k_global );
            
            Dot<AddTo>(a,x,y);
        }
        
    public:
        
        std::string ClassName() const  
        {
            return TOOLS_TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(NRHS)
            +","+TypeName<Scal>
            +","+TypeName<Scal_in>
            +","+TypeName<Scal_out>
            +","+TypeName<Int>
            +","+TypeName<LInt>
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +","+ToString(a_RM)+","+ToString(a_intRM)+","+ToString(a_copy)
            +","+ToString(x_RM)+","+ToString(x_prefetch)
            +","+ToString(y_RM)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
