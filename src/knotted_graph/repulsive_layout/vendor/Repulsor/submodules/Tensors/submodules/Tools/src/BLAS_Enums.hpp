#pragma once

// It is somewhat unorthodox to use the Tensors namespace here, but I need some of these enums also for BLAS 1 routines.

namespace Tools
{
    // cf. CBLAS_TRANSPOSE
    enum class Op: unsigned char
    {
        Id           = 111,
        Trans        = 112,
        ConjTrans    = 113,
        Conj         = 114,
        
        // Non standard
        
        Re           = 115,
        ReTrans      = 116,
        ImTrans      = 117,
        Im           = 118
    };
    
    [[nodiscard]] inline constexpr bool TransposedQ( const Op op )
    {
        return (op == Op::Trans) || (op == Op::ConjTrans) || (op == Op::ReTrans) || (op == Op::ImTrans);
    }
    
    [[nodiscard]] inline constexpr bool NotTransposedQ( const Op op )
    {
        return (op == Op::Id) || (op == Op::Conj) || (op == Op::Re) || (op == Op::Im);
    }
    
    [[nodiscard]] inline constexpr bool ConjugatedQ( const Op op )
    {
        return (op == Op::Conj) || (op == Op::ConjTrans);
    }
    
    [[nodiscard]] inline constexpr Op Transpose( const Op op )
    {
        switch( op )
        {
            case Op::Id:        return Op::Trans;
                
            case Op::Conj:      return Op::ConjTrans;
                
            case Op::Trans:     return Op::Id;
                
            case Op::ConjTrans: return Op::Conj;
                
            case Op::Re:        return Op::ReTrans;
                
            case Op::ReTrans:   return Op::Re;
                
            case Op::Im:        return Op::ImTrans;
                
            case Op::ImTrans:   return Op::Im;
                
            default:
            {
                eprint( "Transpose: Argument op = " + ToString(op) + " is invalid." );
                return Op::Trans;
            }
        }
    }
    
    [[nodiscard]] inline constexpr Op DropTranspose( const Op op )
    {
        switch( op )
        {
            case Op::Id:        return Op::Id;
                
            case Op::Conj:      return Op::Conj;
                
            case Op::Trans:     return Op::Id;
                
            case Op::ConjTrans: return Op::Conj;
                
            case Op::Re:        return Op::Re;
                
            case Op::ReTrans:   return Op::Re;
                
            case Op::Im:        return Op::Im;
                
            case Op::ImTrans:   return Op::Im;
                
            default:
            {
                eprint( "DropTranspose: Argument op = " + ToString(op) + " is invalid." );
                return Op::Id;
            }
        }
    }
    
    [[nodiscard]] inline constexpr Op Conjugate( const Op op )
    {
        switch( op )
        {
            case Op::Id:        return Op::Conj;
                
            case Op::Conj:      return Op::Id;
                
            case Op::Trans:     return Op::ConjTrans;
                
            case Op::ConjTrans: return Op::Trans;
                
            case Op::Re:        return Op::Re;
                
            case Op::ReTrans:   return Op::ReTrans;
                
            case Op::Im:        return Op::Im;
                
            case Op::ImTrans:   return Op::ImTrans;
                
            default:
            {
                eprint( "Conjugate: Argument op = " + ToString(op) + " is invalid." );
                return Op::Conj;
            }
        }
    }
    
    [[nodiscard]] inline std::string ToString( const Op op )
    {
        switch( op )
        {
            case Op::Id:        return "Id";
                
            case Op::Conj:      return "Conj";
                
            case Op::Trans:     return "Trans";
                
            case Op::ConjTrans: return "ConjTrans";
                
            case Op::Re:        return "Re";
                
            case Op::ReTrans:   return "ReTrans";
                
            case Op::Im:        return "Im";
                
            case Op::ImTrans:   return "ImTrans";
                
            default:
            {
                eprint( "ToString: Argument op = " + ToString(op) + " is invalid." );
                return "Id";
            }
        }
    }
    
    // A string that can be used for string code generation.
    [[nodiscard]] inline std::string ToOpString( const Op op )
    {
        switch( op )
        {
            case Op::Id:        return "Tools::Op::Id";
                
            case Op::Conj:      return "Tools::Op::Conj";
                
            case Op::Trans:     return "Tools::Op::Trans";
                
            case Op::ConjTrans: return "Tools::Op::ConjTrans";
                
            case Op::Re:        return "Tools::Op::Re";
                
            case Op::ReTrans:   return "Tools::Op::ReTrans";
                
            case Op::Im:        return "Tools::Op::Im";
                
            case Op::ImTrans:   return "Tools::Op::ImTrans";
                
            default:
            {
                eprint( "ToOpString: Argument op = " + ToString(op) + " is invalid." );
                return "Tools::Op::Id";
            }
        }
    }
    
    // cf. CBLAS_LAYOUT
    enum class Layout : unsigned char
    {
        RowMajor = 101,
        ColMajor = 102
    };
    
    [[nodiscard]] inline constexpr Layout Transpose( const Layout layout )
    {
        if ( layout == Layout::RowMajor )
        {
            return Layout::ColMajor;
        }
        else
        {
            return Layout::RowMajor;
        }
    }
    
    
    [[nodiscard]] inline std::string ToString( const Layout & layout )
    {
        switch( layout )
        {
            case Layout::RowMajor: return "RM";
                
            case Layout::ColMajor: return "CM";
                
            default:
            {
                eprint( "ToString: Argument layout = " + ToString(layout) + " is invalid." );
                return "RM";
            }
        }
    }
    
    // cf. CBLAS_UPLO
    enum class UpLo : unsigned char
    {
        Upper = 121,
        Lower = 122
    };
    
    
    [[nodiscard]] inline std::string ToString( const UpLo & uplo )
    {
        switch( uplo )
        {
            case UpLo::Upper: return "U";
                
            case UpLo::Lower: return "L";
                
            default:
            {
                eprint( "ToString: Argument uplo = " + ToString(uplo) + " is invalid." );
                return "U";
            }
        }
    }
    
    [[nodiscard]] inline constexpr UpLo Transpose( const UpLo uplo )
    {
        switch( uplo )
        {
            case UpLo::Lower:   return UpLo::Upper;
                
            case UpLo::Upper:   return UpLo::Lower;
                
            default:
            {
                eprint( "Transpose: Argument uplo = " + ToString(uplo) + " is invalid." );
                return UpLo::Upper;
            }
        }
    }
    
    // cf. CBLAS_DIAG
    enum class Diag : unsigned char
    {
        NonUnit = 131,
        Unit    = 132
    };
    
    // cf. CBLAS_SIDE
    enum class Side : unsigned char
    {
        Left  = 141,
        Right = 142
    };
    
    constexpr Side Revert( Side s )
    {
        return ( s == Side::Left ) ? Side::Right : Side::Left;
    }

    std::string ToString( Side s )
    {
        return ( s == Side::Left ) ? "Left" : "Right";
    }

    enum class AddTo_T : bool
    {
        True  = true,
        False = false
    };
    
    constexpr AddTo_T AddTo     = AddTo_T::True;
    constexpr AddTo_T Overwrite = AddTo_T::False;
}
