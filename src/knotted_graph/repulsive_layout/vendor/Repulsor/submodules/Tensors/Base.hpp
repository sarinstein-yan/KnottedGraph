#pragma once

#define TENSORS_BASE_H

#include <iostream>
#include <random>
#include <cstring>
#include <complex>

#include "submodules/Tools/Tools.hpp"

namespace Tensors
{
    using namespace Tools;
}

#ifdef LTEMPLATE_H

#include "src/MathematicaTypes.hpp"

#endif

#include "src/Tensor1.hpp"
#include "src/Tensor2.hpp"
#include "src/Tensor3.hpp"
