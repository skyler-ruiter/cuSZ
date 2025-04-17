/**
 * @file l21.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D5965FDA_3E90_4AC4_A53B_8439817D7F1C
#define D5965FDA_3E90_4AC4_A53B_8439817D7F1C

#include <stdint.h>

#include "cusz/type.h"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#define PROPER_EB f8
#elif defined(PSZ_USE_1API)
#define PROPER_EB f4
#endif

namespace psz::module {

template <typename T, typename Eq>
int GPU_PROTO_c_lorenzo_nd_with_outlier(
    T* const in_data, std::array<size_t, 3> const data_len3, Eq* const out_eq, void* out_outlier,
    f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream);

template <typename T, typename Eq>
int GPU_PROTO_x_lorenzo_nd(
    Eq* in_eq, T* in_outlier, T* out_data, std::array<size_t, 3> const data_len3, f8 const ebx2,
    f8 const ebx2_r, int const radius, void* stream);

}  // namespace psz::module

#endif /* D5965FDA_3E90_4AC4_A53B_8439817D7F1C */
