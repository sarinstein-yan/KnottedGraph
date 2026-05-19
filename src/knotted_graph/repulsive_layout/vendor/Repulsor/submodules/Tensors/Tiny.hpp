#pragma once

namespace Tensors
{
    // Never do `using namespace Tiny`. Chaos will break loose.
    namespace Tiny
    {
        static constexpr Op O_Id        = Op::Id;
        static constexpr Op O_Trans     = Op::Trans;
        static constexpr Op O_Conj      = Op::Conj;
        static constexpr Op O_ConjTrans = Op::ConjTrans;
        
        using Flag = Scalar::Flag;
        
        static constexpr Flag F_Plus    = Flag::Plus;
        static constexpr Flag F_Minus   = Flag::Minus;
        static constexpr Flag F_Gen     = Flag::Generic;
        static constexpr Flag F_Zero    = Flag::Zero;
    }
}

#include "src/Tiny/fixed_dot_mm_naive.hpp"
//#include "src/Tiny/fixed_dot_mm_vec.hpp"

#if ( __has_attribute(matrix_type) )
    #include "src/Tiny/fixed_dot_mm_clang.hpp"
#endif

#include "src/Tiny/fixed_dot_mm.hpp"

#include "src/Tiny/Vector.hpp"
#include "src/Tiny/VectorList.hpp"
#include "src/Tiny/VectorList_AoS.hpp"

#include "src/Tiny/Matrix.hpp"
#include "src/Tiny/UpperTriangularMatrix.hpp"
#include "src/Tiny/SelfAdjointTridiagonalMatrix.hpp"
#include "src/Tiny/SelfAdjointMatrix.hpp"
#include "src/Tiny/MatrixList.hpp"
#include "src/Tiny/MatrixList_AoS.hpp"

#include "src/Tiny/gemm.hpp"
#include "src/Tiny/trsm.hpp"
