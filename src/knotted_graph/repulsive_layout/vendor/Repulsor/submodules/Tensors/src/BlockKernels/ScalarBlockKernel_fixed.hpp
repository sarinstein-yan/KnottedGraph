#pragma once

#define CLASS ScalarBlockKernel_fixed

#define BASE  BlockKernel_fixed<                            \
    ROWS_,COLS_,NRHS_,fixed,                           \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_, LInt_,                                            \
    alpha_flag, beta_flag,                                  \
    x_RM, x_intRM, x_copy, x_prefetch,                      \
    y_RM, y_intRM,                                          \
    use_fma                                                 \
>

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
    class CLASS : public BASE
    {
        
        static_assert( ROWS_ == COLS_ );
        
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;
        using Int        = Int_;
        using LInt       = LInt_;

        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::NRHS;
        
        using BASE::FMA;
        
        static constexpr LInt BLOCK_NNZ = 1;
        
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
        
    public:
        
        explicit CLASS( mptr<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            cptr<Scal> A_,
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
        
        static constexpr LInt NonzeroCount()
        {
            return BLOCK_NNZ;
        }
                
        TOOLS_FORCE_INLINE void TransposeBlock( const LInt from, const LInt to ) const
        {
            A[BLOCK_NNZ * to] = A[BLOCK_NNZ * from];
        }
        
        TOOLS_FORCE_INLINE void ApplyBlock( const LInt k_global, const Int j_global )
        {
            TOOLS_MAKE_FP_FAST()
            
            ReadX( j_global );

            const Scal a = A_const[BLOCK_NNZ * k_global];
            
            TOOLS_LOOP_UNROLL_FULL
            for( Int j = 0; j < COLS; ++j )
            {
                TOOLS_LOOP_UNROLL_FULL
                for( Int k = 0; k < NRHS; ++k )
                {
                    FMA( a, get_x(j,k), get_y(j,k) );
                }
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
            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)
            +","+ToString(use_fma)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
