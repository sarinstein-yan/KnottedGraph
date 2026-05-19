#pragma once

// Basically does what BLAS 1 does, but not strictly following the syntax.


#include "BLAS_1/copy_buffer.hpp"
#include "BLAS_1/move_buffer.hpp"
#include "BLAS_1/swap_buffers.hpp"
#include "BLAS_1/zerofy_buffer.hpp"
#include "BLAS_1/fill_buffer.hpp"

#include "BLAS_1/zip_buffers.hpp"

#include "BLAS_1/modify_buffer.hpp"
#include "BLAS_1/scale_buffer.hpp"
//#include "BLAS_1/conjugate_buffer.hpp" // obsolete
#include "BLAS_1/combine_buffers.hpp"

#include "BLAS_1/add_to_buffer.hpp"

#include "BLAS_1/dot_buffers.hpp"

#include "BLAS_1/iamax_buffer.hpp"

#include "BLAS_1/total_buffer.hpp"
#include "BLAS_1/norm_1.hpp"
#include "BLAS_1/norm_2.hpp"
#include "BLAS_1/norm_max.hpp"


#include "BLAS_1/iota_buffer.hpp"
#include "BLAS_1/minmax_buffer.hpp"
#include "BLAS_1/elementwise_minmax.hpp"

#include "BLAS_1/scatter_read_combine.hpp"
#include "BLAS_1/combine_scatter_write.hpp"

#include "BLAS_1/count_nans_in_buffer.hpp"

#include "BLAS_1/Find.hpp"
#include "BLAS_1/buffers_equalQ.hpp"

#include "BLAS_1/rotate_buffer.hpp"
