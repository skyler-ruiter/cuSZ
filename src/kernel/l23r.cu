/**
 * @file l32r_cu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "cusz/type.h"
#include "detail/l23r.inl"
#include "kernel/l23r.hh"
#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq, bool ZigZag>
pszerror psz_comp_l23r(
    T* const data, dim3 const len3, f8 const eb, int const radius,
    Eq* const eq, void* _outlier, f4* time_elapsed, void* stream,
    uint32_t* hist)
{
  static_assert(
      std::is_same<Eq, u4>::value or std::is_same<Eq, uint16_t>::value or
          std::is_same<Eq, uint8_t>::value,
      "Eq must be unsigned integer that is less than or equal to 4 bytes.");

  auto div3 = [](dim3 len, dim3 tile) {
    return dim3(
        (len.x - 1) / tile.x + 1, (len.y - 1) / tile.y + 1,
        (len.z - 1) / tile.z + 1);
  };
  auto mult3 = [](dim3 len, dim3 factor) {
    return dim3(len.x * factor.x, len.y * factor.y, len.z * factor.z);
  };

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  using Compact = CompactGpuDram<T>;
  auto ot = (Compact*)_outlier;

  auto d = ndim();

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    /*
    { // baseline
      constexpr auto Tile1D = 256, Seq1D = 4, Block1D = 64;
      auto Grid1D = div3(len3, Tile1D);
      psz::rolling::c_lorenzo_1d1l<T, false, Eq, T, Tile1D, Seq1D>
          <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
              data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
              ot->num());
    }
     */
    {  // v2: add rep to baseline
      constexpr auto Tile1D = 256, Seq1D = 4, Block1D = 64, Rep = 16;
      auto Grid1D = div3(len3, Tile1D * Rep);

      /*
        psz::rolling::c_lorenzo_1d1l_v2<T, false, Eq, T, Tile1D, Seq1D>
            <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
                data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
                ot->num(), Rep);

        // v3: add hist to v2
        psz::rolling::c_lorenzo_1d1l_v3<T, false, Eq, T, Tile1D, Seq1D>
            <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
                data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
                ot->num(), Rep, hist);
                */

      // v4 <- v3 + less shmem access
      psz::rolling::c_lorenzo_1d1l_v4<T, false, Eq, T, Tile1D, Seq1D>
          <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
              data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
              ot->num(), Rep, hist);
    }
  }
  else if (d == 2) {
    // {  // baseline
    //   constexpr auto Tile2D = dim3(16, 16, 1), Block2D = dim3(16, 2, 1);
    //   auto Grid2D = div3(len3, Tile2D);
    //   psz::rolling::c_lorenzo_2d1l<T, false, Eq, T>
    //       <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
    //           data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
    //           ot->num());
    // }

    // {  //  v2: add rep to baseline
    //   constexpr auto Rep2 = dim3(2, 4), Tile2D = dim3(16, 16, 1),
    //                  Shard2D = mult3(Tile2D, Rep2);
    //   constexpr auto Block2D = dim3(16, 2, 1);
    //   auto Grid2D = div3(len3, Shard2D);

    //   psz::rolling::c_lorenzo_2d1l_v2<T, false, Eq, T>
    //       <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
    //           data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
    //           ot->num(), Rep2);
    // }
    {
      // v3 & v4
      // constexpr auto Rep2 = dim3(2, 4), Tile2D = dim3(16, 16, 1),
      //                Shard2D = mult3(Tile2D, Rep2);
      // constexpr auto Block2D = dim3(32, 1, 1);
      // auto Grid2D = div3(len3, Shard2D);

      // // v3: add linearized thread id to v2
      // psz::rolling::c_lorenzo_2d1l_v3<T, false, Eq, T>
      //     <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
      //         data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
      //         ot->num(), Rep2);

      constexpr auto Rep2 = dim3(2, 4), Tile2D = dim3(16, 16, 1),
                     Shard2D = mult3(Tile2D, Rep2);
      constexpr auto Block2D = dim3(32, 1, 1);
      auto Grid2D = div3(len3, Shard2D);

      // v4: add dram hist to v3
      psz::rolling::c_lorenzo_2d1l_v4<T, false, Eq, T>
          <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
              data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
              ot->num(), Rep2, hist);
    }
  }
  else if (d == 3) {
    // {  // baseline
    //   constexpr auto Tile3D = dim3(32, 8, 8);
    //   constexpr auto Block3D = dim3(32, 8, 1);
    //   auto Grid3D = div3(len3, Tile3D);
    //   psz::rolling::c_lorenzo_3d1l<T, false, Eq, T>
    //       <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
    //           data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
    //           ot->num());
    // }

    {  // v2: add rep to baseline
      constexpr auto Tile3D = dim3(32, 8, 8);
      constexpr auto Block3D = dim3(32, 8, 1);
      constexpr auto Rep3 = dim3(1, 2, 2);
      constexpr auto Shard3D = mult3(Tile3D, Rep3);
      auto Grid3D = div3(len3, Shard3D);
      // psz::rolling::c_lorenzo_3d1l_v2<T, false, Eq, T>
      //     <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
      //         data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
      //         ot->num(), Rep3);

      // v3: simplification of v2
      // psz::rolling::c_lorenzo_3d1l_v3<T, false, Eq, T>
      //     <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
      //         data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
      //         ot->num(), Rep3);

      // v4: add hist to v3
      psz::rolling::c_lorenzo_3d1l_v4<T, false, Eq, T>
          <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
              data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
              ot->num(), Rep3, hist);
    }
  }

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#define INIT(T, E, ZIGZAG)                                           \
  template pszerror psz_comp_l23r<T, E, ZIGZAG>(                     \
      T* const data, dim3 const len3, f8 const eb, int const radius, \
      E* const eq, void* _outlier, f4* time_elapsed, void* stream,   \
      uint32_t* hist);

INIT(f4, u4, false)
INIT(f4, u4, true)
INIT(f8, u4, false)
INIT(f8, u4, true)

#undef INIT