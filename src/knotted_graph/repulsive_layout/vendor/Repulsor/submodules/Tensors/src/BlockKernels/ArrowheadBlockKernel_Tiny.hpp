#pragma once

#define CLASS ArrowheadBlockKernel_Tiny

#define BASE  BlockKernel_Tiny<                             \
    ROWS_,COLS_,NRHS_,                                      \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_, LInt_,                                            \
    alpha_flag, beta_flag,                                  \
    x_RM, x_prefetch,                                       \
    y_RM                                                    \
>

//template<
//    int ROWS_, int COLS_, int NRHS_,
//    typename Scal_, IntQ Int_, typename Scal_in_, typename Scal_out_,
//    Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
//    bool a_copy = true,
//    bool x_RM  = true, bool x_prefetch = true,
//    bool y_RM  = true
//>
namespace Tensors
{
    template<
        int ROWS_, int COLS_, int NRHS_,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        IntQ Int_, IntQ LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        bool a_copy,
        bool x_RM, bool x_prefetch,
        bool y_RM
    >
    class CLASS : public BASE
    {
        static_assert( ROWS_ == COLS_ );
        
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;
        using Int      = Int_;
        using LInt     = LInt_;

        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::NRHS;
        
        static constexpr LInt BLOCK_NNZ = COLS + ROWS - 1;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::y;
        
        using BASE::ReadX;
        
        const Scal * restrict a_from = nullptr;
        
        Tiny::Vector<BLOCK_NNZ,Scal,Int> a;
        
    public:
        
        explicit CLASS( mptr<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            cptr<Scal> A_,
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            const Int nrhs_
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
            cptr<Scal> a_from_ = &A[BLOCK_NNZ * from];
            mptr<Scal> a_to_   = &A[BLOCK_NNZ * to  ];
            
            a_to_[0] = a_from_[0];
            
            for( Int i = 1; i < ROWS; ++i )
            {
                a_to_[       i] = a_from_[ROWS-1+i];
                a_to_[ROWS-1+i] = a_from_[       i];
            }
        }
        
        TOOLS_FORCE_INLINE void ReadA( const LInt k_global )
        {
            // Read matrix.
            if constexpr ( a_copy )
            {
                a.Read( &A_const[BLOCK_NNZ * k_global] );
            }
            else
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
            }
        }
        
        TOOLS_FORCE_INLINE cref<Scal> get_a( const Int l )
        {
            if constexpr ( a_copy )
            {
                return a[l];
            }
            else
            {
                return a_from[l];
            }
        }
        
        
        TOOLS_FORCE_INLINE void ApplyBlock( const LInt k_global, const Int j_global )
        {
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            ReadX( j_global );
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            
            ReadA( k_global );
            
            /*
             *     /                                                                  \
             *     |   get_a(0)          get_a(1)       get_a(2)      get_a(COLS-1)   |
             *     |                                                                  |
             *     |   get_a(COLS)          0              0              0           |
             *     |                                                                  |
             *     |   get_a(COLS+1)        0              0              0           |
             *     |                                                                  |
             *     |   get_a(ROWS+COLS-2)   0              0              0           |
             *     \                                                                  /
             */
            
            for( Int k = 0; k < NRHS; ++k )
            {
                y[0][k] += get_a(0) * x[0][k];
                
                for( Int j = 1; j < COLS; ++j )
                {
                    y[0][k] += get_a(j) * x[j][k];
                }
                
                for( Int i = 1; i < ROWS; ++i )
                {
                    y[i][k] += get_a(COLS-1+i) * x[0][k];
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
            +","+TypeName<Scal>
            +","+TypeName<Scal_in>
            +","+TypeName<Scal_out>
            +","+TypeName<Int>
            +","+TypeName<LInt>
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +","+ToString(a_copy)
            +","+ToString(x_RM)
            +","+ToString(x_prefetch)
            +","+ToString(y_RM)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
