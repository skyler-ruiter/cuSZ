/**
 * @file l23r.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AAC905A6_6314_4E1E_B5CD_BBBA9005A448
#define AAC905A6_6314_4E1E_B5CD_BBBA9005A448

#include <math.h>
#include <stdint.h>

#include <type_traits>

#include "cusz/suint.hh"
#include "mem/compact.hh"
#include "port.hh"

#define SETUP_ZIGZAG                                                         \
  using EqUint = typename psz::typing::UInt<sizeof(Eq)>::T;                  \
  using EqInt = typename psz::typing::Int<sizeof(Eq)>::T;                    \
  static_assert(                                                             \
      std::is_same<Eq, EqUint>::value, "Eq must be unsigned integer type."); \
  auto posneg_encode = [](EqInt x) -> EqUint {                               \
    return (2 * (x)) ^ ((x) >> (sizeof(Eq) * 8 - 1));                        \
  };

namespace psz {
namespace rolling {

template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    int TileDim = 256, int Seq = 8, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t>
__global__ void c_lorenzo_1d1l(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  constexpr auto NumThreads = TileDim / Seq;

  SETUP_ZIGZAG;

  __shared__ union {
    T data[TileDim];
    EqUint eq_uint[TileDim];
  } s;

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto id_base = blockIdx.x * TileDim;

// dram.data to shmem.data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < len3.x)
      s.data[threadIdx.x + ix * NumThreads] = round(data[id] * ebx2_r);
  }
  __syncthreads();

// shmem.data to private.data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++)
    thp_data(ix) = s.data[threadIdx.x * Seq + ix];
  if (threadIdx.x > 0)
    prev() = s.data[threadIdx.x * Seq - 1];  // from last thread
  __syncthreads();

  /* from here on, s.data is no longer used, therefore, union */

  // quantize & write back to shmem.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    T delta = thp_data(ix) - thp_data(ix - 1);
    bool quantizable = fabs(delta) < radius;
    T candidate = ZigZag ? delta : delta + radius;
    // otherwise, need to reset shared memory (to 0)
    if (ZigZag)
      s.eq_uint[ix + threadIdx.x * Seq] =
          posneg_encode(quantizable * static_cast<EqInt>(candidate));
    else
      s.eq_uint[ix + threadIdx.x * Seq] =
          quantizable * static_cast<EqUint>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(cn, 1);
      cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
      cval[cur_idx] = candidate;
    }
  }
  __syncthreads();

// write from shmem.eq to dram.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < len3.x) eq[id] = s.eq_uint[threadIdx.x + ix * NumThreads];
  }

  // end of kernel
}

// with repeat
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    int TileDim = 256, int Seq = 8, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t>
__global__ void c_lorenzo_1d1l_v2(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, int repeat)
{
  constexpr auto NumThreads = TileDim / Seq;

  SETUP_ZIGZAG;

  __shared__ union {
    T data[TileDim];
    EqUint eq_uint[TileDim];
  } s;

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  for (auto r = 0; r < repeat; r++) {
    auto id_base = blockIdx.x * (repeat * TileDim) + r * TileDim;

// dram.data to shmem.data
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto id = id_base + threadIdx.x + ix * NumThreads;
      if (id < len3.x)
        s.data[threadIdx.x + ix * NumThreads] = round(data[id] * ebx2_r);
    }
    __syncthreads();

// shmem.data to private.data
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++)
      thp_data(ix) = s.data[threadIdx.x * Seq + ix];
    if (threadIdx.x > 0)
      prev() = s.data[threadIdx.x * Seq - 1];  // from last thread
    __syncthreads();

    /* from here on, s.data is no longer used, therefore, union */

    // quantize & write back to shmem.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      T delta = thp_data(ix) - thp_data(ix - 1);
      bool quantizable = fabs(delta) < radius;
      T candidate = ZigZag ? delta : delta + radius;
      // otherwise, need to reset shared memory (to 0)
      if (ZigZag)
        s.eq_uint[ix + threadIdx.x * Seq] =
            posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        s.eq_uint[ix + threadIdx.x * Seq] =
            quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
        cval[cur_idx] = candidate;
      }
    }
    __syncthreads();

// write from shmem.eq to dram.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto id = id_base + threadIdx.x + ix * NumThreads;
      if (id < len3.x) eq[id] = s.eq_uint[threadIdx.x + ix * NumThreads];
    }
  }
  // end of kernel
}

// with repeat + local histsp
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    int TileDim = 256, int Seq = 8, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t,
    int SmallRadius = 128, int TinyRadius = 5>
__global__ void c_lorenzo_1d1l_v3(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, int repeat,
    uint32_t* dram_hist)
{
  constexpr auto NumThreads = TileDim / Seq;
  constexpr auto K = TinyRadius;
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;  // K = 5, R = 2

  SETUP_ZIGZAG;

  __shared__ struct {
    union {
      T data[TileDim];
      EqUint eq_uint[TileDim];
    };
    uint32_t hist[SmallRadius * 2];
  } s;

  T _thp_data[Seq + 1] = {0};
  uint32_t p_hist[K] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  /*  */
  for (auto i = threadIdx.x; i < SmallRadius * 2; i += blockDim.x)
    s.hist[i] = 0;
  __syncthreads();

  for (auto r = 0; r < repeat; r++) {
    auto id_base = blockIdx.x * (repeat * TileDim) + r * TileDim;

// dram.data to shmem.data
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto id = id_base + threadIdx.x + ix * NumThreads;
      if (id < len3.x)
        s.data[threadIdx.x + ix * NumThreads] = round(data[id] * ebx2_r);
    }
    __syncthreads();

// shmem.data to private.data
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++)
      thp_data(ix) = s.data[threadIdx.x * Seq + ix];
    if (threadIdx.x > 0)
      prev() = s.data[threadIdx.x * Seq - 1];  // from last thread
    __syncthreads();

    /* from here on, s.data is no longer used, therefore, union */

    // quantize & write back to shmem.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      T delta = thp_data(ix) - thp_data(ix - 1);
      bool quantizable = fabs(delta) < radius;
      T candidate = ZigZag ? delta : delta + radius;
      // otherwise, need to reset shared memory (to 0)
      if (ZigZag)
        s.eq_uint[ix + threadIdx.x * Seq] =
            posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        s.eq_uint[ix + threadIdx.x * Seq] =
            quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
        cval[cur_idx] = candidate;
      }
    }
    __syncthreads();

// [TODO] special treatment of outlier indication: 0

// write from shmem.eq to dram.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      // sequantiality loop end
      auto id = id_base + threadIdx.x + ix * NumThreads;

      // boundary check start
      if (id < len3.x) {
        auto eq_val = s.eq_uint[threadIdx.x + ix * NumThreads];  // e.g., 512

        eq[id] = eq_val;

        int sym = eq_val - radius;  // e.g., 512 - 512 = 0

        if (abs(sym) < SmallRadius) {  // e.g., |sym| < 128, -128 < sym < 128
          if (2 * abs(sym) < K) {
            // -2, -1, 0, 1, 2 -> sym + R = 0, 1, 2, 3, 4
            //  4   2  0  2  4 <- 2 * abs(sym)
            p_hist[sym + R] += 1;  // more possible
          }
          else {
            // resume the original input
            atomicAdd(&s.hist[sym + SmallRadius], 1);  // less possible
          }
        }
        else {
          atomicAdd(&dram_hist[eq_val], 1);  // less possible
        }

        // (memory hierachy for hist) check end
      }
      // boundary check end
    }
    // sequantiality loop end
  }
  // end of rep-loop

  /* write out hist */
  for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      auto n = __shfl_up_sync(0xffffffff, sum, d);
      if (threadIdx.x % 32 >= d) sum += n;
    }
  }

  for (auto i = 0; i < K; i++)
    if (threadIdx.x % 32 == 31)
      atomicAdd(&s.hist[(int)SmallRadius + i - R], p_hist[i]);
  __syncthreads();

  for (auto i = threadIdx.x; i < 2 * SmallRadius; i += blockDim.x)
    atomicAdd(dram_hist + (i - SmallRadius + radius), s.hist[i]);
  __syncthreads();

  // end of kernel
}

// with repeat + local histsp
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    int TileDim = 256, int Seq = 8, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t,
    int SmallRadius = 128, int TinyRadius = 5>
__global__ void c_lorenzo_1d1l_v4(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, int repeat,
    uint32_t* dram_hist)
{
  constexpr auto NumThreads = TileDim / Seq;
  constexpr auto K = TinyRadius;
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;  // K = 5, R = 2

  SETUP_ZIGZAG;

  __shared__ struct {
    union {
      T data[TileDim];
      EqUint eq_uint[TileDim];
    };
    uint32_t hist[SmallRadius * 2];
  } s;

  T _thp_data[Seq + 1] = {0};
  uint32_t p_hist[K] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  /*  */
  for (auto i = threadIdx.x; i < SmallRadius * 2; i += blockDim.x)
    s.hist[i] = 0;
  __syncthreads();

  for (auto r = 0; r < repeat; r++) {
    auto id_base = blockIdx.x * (repeat * TileDim) + r * TileDim;

// dram.data to shmem.data
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto id = id_base + threadIdx.x + ix * NumThreads;
      if (id < len3.x)
        s.data[threadIdx.x + ix * NumThreads] = round(data[id] * ebx2_r);
    }
    __syncthreads();

// shmem.data to private.data
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++)
      thp_data(ix) = s.data[threadIdx.x * Seq + ix];
    if (threadIdx.x > 0)
      prev() = s.data[threadIdx.x * Seq - 1];  // from last thread
    __syncthreads();

    /* from here on, s.data is no longer used, therefore, union */

    // quantize & write back to shmem.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      T delta = thp_data(ix) - thp_data(ix - 1);
      bool quantizable = fabs(delta) < radius;

      T candidate = ZigZag ? delta : delta + radius;
      // otherwise, need to reset shared memory (to 0)

      if (abs(delta) < SmallRadius) {  // e.g., |delta| < 128
        if (2 * abs(delta) < K) {
          // -2, -1, 0, 1, 2 -> sym + R = 0, 1, 2, 3, 4
          //  4   2  0  2  4 <- 2 * abs(delta)
          p_hist[(int)delta + R] += 1;  // more possible
        }
        else {
          // resume the original input
          atomicAdd(&s.hist[(int)delta + SmallRadius], 1);  // less possible
        }
      }
      else {
        // less possible
        if (ZigZag)
          atomicAdd(&dram_hist[((EqInt)candidate) * quantizable], 1);
        else
          atomicAdd(&dram_hist[((EqUint)candidate) * quantizable], 1);
      }
      // [TODO] leave room for zigzag

      if (ZigZag)
        s.eq_uint[ix + threadIdx.x * Seq] =
            posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        s.eq_uint[ix + threadIdx.x * Seq] =
            quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
        cval[cur_idx] = candidate;
      }
    }
    __syncthreads();

// [TODO] special treatment of outlier indication: 0

// write from shmem.eq to dram.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      // sequantiality loop end
      auto id = id_base + threadIdx.x + ix * NumThreads;

      // boundary check start
      if (id < len3.x) {
        eq[id] = s.eq_uint[threadIdx.x + ix * NumThreads];  // e.g., 512

        // (memory hierachy for hist) check end
      }
      // boundary check end
    }
    // sequantiality loop end
  }
  // end of rep-loop

  /* write out hist */
  for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      auto n = __shfl_up_sync(0xffffffff, sum, d);
      if (threadIdx.x % 32 >= d) sum += n;
    }
  }

  for (auto i = 0; i < K; i++)
    if (threadIdx.x % 32 == 31)
      atomicAdd(&s.hist[(int)SmallRadius + i - R], p_hist[i]);
  __syncthreads();

  for (auto i = threadIdx.x; i < 2 * SmallRadius; i += blockDim.x)
    atomicAdd(dram_hist + (i - SmallRadius + radius), s.hist[i]);
  __syncthreads();

  // end of kernel
}

template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
__global__ void c_lorenzo_2d1l(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;

  SETUP_ZIGZAG;

  // NW  N       first el <- 0
  //  W  center
  T center[Yseq + 1] = {0};
  // auto prev = [&]() -> T& { return _center[0]; };
  // auto center = [&](auto i) -> T& { return _center[i + 1]; };
  // auto last = [&]() -> T& { return _center[Yseq]; };

  // BDX == TileDim == 16, BDY * Yseq = TileDim == 16
  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
  auto g_id = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

  // use a warp as two half-warps
  // block_dim = (16, 2, 1) makes a full warp internally

// read to private.data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < len3.x and giy_base + iy < len3.y)
      center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
  }
  // same-warp, next-16
  auto tmp = __shfl_up_sync(0xffffffff, center[Yseq], 16, 32);
  if (threadIdx.y == 1) center[0] = tmp;

// prediction (apply Lorenzo filter)
#pragma unroll
  for (auto i = Yseq; i > 0; i--) {
    // with center[i-1] intact in this iteration
    center[i] -= center[i - 1];
    // within a halfwarp (32/2)
    auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
    if (threadIdx.x > 0) center[i] -= west;  // delta
  }
  __syncthreads();

#pragma unroll
  for (auto i = 1; i < Yseq + 1; i++) {
    auto gid = g_id(i - 1);

    if (gix < len3.x and giy_base + (i - 1) < len3.y) {
      bool quantizable = fabs(center[i]) < radius;
      T candidate = ZigZag ? center[i] : center[i] + radius;
      if (ZigZag)
        eq[gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        eq[gid] = quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = gid;
        cval[cur_idx] = candidate;
      }
    }
  }

  // end of kernel
}

// add rep
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
__global__ void c_lorenzo_2d1l_v2(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, dim3 repeat2)
{
  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;

  SETUP_ZIGZAG;

  // NW  N       first el <- 0
  //  W  center
  T center[Yseq + 1] = {0};
  // auto prev = [&]() -> T& { return _center[0]; };
  // auto center = [&](auto i) -> T& { return _center[i + 1]; };
  // auto last = [&]() -> T& { return _center[Yseq]; };
  // BDX == TileDim == 16, BDY * Yseq = TileDim == 16

  for (auto ry = 0; ry < repeat2.y; ry++) {
    for (auto rx = 0; rx < repeat2.x; rx++) {
      auto gix =
          blockIdx.x * (TileDim * repeat2.x) + (TileDim * rx) + threadIdx.x;
      auto giy_base = blockIdx.y * (TileDim * repeat2.y) + (TileDim * ry) +
                      threadIdx.y * Yseq;
      auto g_id = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

      // use a warp as two half-warps
      // block_dim = (16, 2, 1) makes a full warp internally

// read to private.data (center)
#pragma unroll
      for (auto iy = 0; iy < Yseq; iy++) {
        if (gix < len3.x and giy_base + iy < len3.y)
          center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
      }
      // same-warp, next-16
      auto tmp = __shfl_up_sync(0xffffffff, center[Yseq], 16, 32);
      if (threadIdx.y == 1) center[0] = tmp;

// prediction (apply Lorenzo filter)
#pragma unroll
      for (auto i = Yseq; i > 0; i--) {
        // with center[i-1] intact in this iteration
        center[i] -= center[i - 1];
        // within a halfwarp (32/2)
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (threadIdx.x > 0) center[i] -= west;  // delta
      }  // end of sequantiality loop for prediction
      __syncthreads();

#pragma unroll
      for (auto i = 1; i < Yseq + 1; i++) {
        auto gid = g_id(i - 1);

        if (gix < len3.x and giy_base + (i - 1) < len3.y) {
          bool quantizable = fabs(center[i]) < radius;
          T candidate = ZigZag ? center[i] : center[i] + radius;
          if (ZigZag)
            eq[gid] =
                posneg_encode(quantizable * static_cast<EqInt>(candidate));
          else
            eq[gid] = quantizable * static_cast<EqUint>(candidate);

          // stream compaction
          if (not quantizable) {
            auto cur_idx = atomicAdd(cn, 1);
            cidx[cur_idx] = gid;
            cval[cur_idx] = candidate;
          }
          // end of writing outlier
        }
        // end of boundary check
      }
      // end of sequantiality loop for quantization
    }
    // end of rep-loop (inner)
  }
  // end of rep-loop (outer)

  // end of kernel
}

// v2 <- thread id change
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
__global__ void c_lorenzo_2d1l_v3(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, dim3 repeat2)
{
  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;

  SETUP_ZIGZAG;

  // NW  N       first el <- 0
  //  W  center
  T center[Yseq + 1] = {0};
  // auto prev = [&]() -> T& { return _center[0]; };
  // auto center = [&](auto i) -> T& { return _center[i + 1]; };
  // auto last = [&]() -> T& { return _center[Yseq]; };
  // BDX == TileDim == 16, BDY * Yseq = TileDim == 16

  auto thx = [&]() { return threadIdx.x % 16; };
  auto thy = [&]() { return threadIdx.x / 16; };

  for (auto ry = 0; ry < repeat2.y; ry++) {
    for (auto rx = 0; rx < repeat2.x; rx++) {
      auto gix = blockIdx.x * (TileDim * repeat2.x) + (TileDim * rx) + thx();
      auto giy_base =
          blockIdx.y * (TileDim * repeat2.y) + (TileDim * ry) + thy() * Yseq;
      auto g_id = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

      // use a warp as two half-warps
      // block_dim = (16, 2, 1) makes a full warp internally

// read to private.data (center)
#pragma unroll
      for (auto iy = 0; iy < Yseq; iy++) {
        if (gix < len3.x and giy_base + iy < len3.y)
          center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
      }
      // same-warp, next-16
      auto tmp = __shfl_up_sync(0xffffffff, center[Yseq], 16, 32);
      if (thy() == 1) center[0] = tmp;

// prediction (apply Lorenzo filter)
#pragma unroll
      for (auto i = Yseq; i > 0; i--) {
        // with center[i-1] intact in this iteration
        center[i] -= center[i - 1];
        // within a halfwarp (32/2)
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (thx() > 0) center[i] -= west;  // delta
      }  // end of sequantiality loop for prediction
      __syncthreads();

#pragma unroll
      for (auto i = 1; i < Yseq + 1; i++) {
        auto gid = g_id(i - 1);

        if (gix < len3.x and giy_base + (i - 1) < len3.y) {
          bool quantizable = fabs(center[i]) < radius;
          T candidate = ZigZag ? center[i] : center[i] + radius;
          if (ZigZag)
            eq[gid] =
                posneg_encode(quantizable * static_cast<EqInt>(candidate));
          else
            eq[gid] = quantizable * static_cast<EqUint>(candidate);

          // stream compaction
          if (not quantizable) {
            auto cur_idx = atomicAdd(cn, 1);
            cidx[cur_idx] = gid;
            cval[cur_idx] = candidate;
          }
          // end of writing outlier
        }
        // end of boundary check
      }
      // end of sequantiality loop for quantization
    }
    // end of rep-loop (inner)
  }
  // end of rep-loop (outer)

  // end of kernel
}

// v3 <- add hist
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, int SmallRadius = 128, int TinyRadius = 1>
__global__ void c_lorenzo_2d1l_v4(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, dim3 repeat2,
    uint32_t* dram_hist)
{
  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;
  constexpr auto K = TinyRadius;
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;  // K = 5, R = 2

  SETUP_ZIGZAG;

  __shared__ struct {
    uint32_t hist[SmallRadius * 2];
  } s;

  auto thx = [&]() { return threadIdx.x % 16; };
  auto thy = [&]() { return threadIdx.x / 16; };
  auto th_linear = [&]() { return threadIdx.x; };
  auto nworker = [&]() { return blockDim.x; };
  // NW  N       first el <- 0
  //  W  center
  uint32_t p_hist[K] = {0};
  T center[Yseq + 1] = {0};
  // auto prev = [&]() -> T& { return _center[0]; };
  // auto center = [&](auto i) -> T& { return _center[i + 1]; };
  // auto last = [&]() -> T& { return _center[Yseq]; };
  // BDX == TileDim == 16, BDY * Yseq = TileDim == 16

  /* clear s_hist buffer */
  for (auto i = th_linear(); i < SmallRadius * 2; i += nworker())
    s.hist[i] = 0;
  __syncthreads();

  for (auto ry = 0; ry < repeat2.y; ry++) {
    for (auto rx = 0; rx < repeat2.x; rx++) {
      auto gix = blockIdx.x * (TileDim * repeat2.x) + (TileDim * rx) + thx();
      auto giy_base =
          blockIdx.y * (TileDim * repeat2.y) + (TileDim * ry) + thy() * Yseq;
      auto g_id = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

      // use a warp as two half-warps
      // block_dim = (16, 2, 1) makes a full warp internally

// read to private.data (center)
#pragma unroll
      for (auto iy = 0; iy < Yseq; iy++) {
        if (gix < len3.x and giy_base + iy < len3.y)
          center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
      }
      // same-warp, next-16
      auto tmp = __shfl_up_sync(0xffffffff, center[Yseq], 16, 32);
      if (thy() == 1) center[0] = tmp;

// prediction (apply Lorenzo filter)
#pragma unroll
      for (auto i = Yseq; i > 0; i--) {
        // with center[i-1] intact in this iteration
        center[i] -= center[i - 1];
        // within a halfwarp (32/2)
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (thx() > 0) center[i] -= west;  // delta
      }  // end of sequantiality loop for prediction
      __syncthreads();

#pragma unroll
      for (auto i = 1; i < Yseq + 1; i++) {
        auto gid = g_id(i - 1);

        if (gix < len3.x and giy_base + (i - 1) < len3.y) {
          bool quantizable = fabs(center[i]) < radius;
          T candidate = ZigZag ? center[i] : center[i] + radius;

          if (abs(center[i]) < SmallRadius) {  // e.g., |delta| < 128
            if (2 * abs(center[i]) < K) {
              // -2, -1, 0, 1, 2 -> sym + R = 0, 1, 2, 3, 4
              //  4   2  0  2  4 <- 2 * abs(delta)
              p_hist[(int)center[i] + R] += 1;  // more possible
            }
            else {
              // resume the original input
              atomicAdd(
                  &s.hist[(int)center[i] + SmallRadius], 1);  // less possible
            }
          }
          else {
            // less possible
            if (ZigZag)
              atomicAdd(&dram_hist[((EqInt)candidate) * quantizable], 1);
            else
              atomicAdd(&dram_hist[((EqUint)candidate) * quantizable], 1);
          }

          if (ZigZag)
            eq[gid] =
                posneg_encode(quantizable * static_cast<EqInt>(candidate));
          else
            eq[gid] = quantizable * static_cast<EqUint>(candidate);

          // stream compaction
          if (not quantizable) {
            auto cur_idx = atomicAdd(cn, 1);
            cidx[cur_idx] = gid;
            cval[cur_idx] = candidate;
          }
          // end of writing outlier
        }
        // end of boundary check
      }
      // end of sequantiality loop for quantization
    }
    // end of rep-loop (inner)
  }
  // end of rep-loop (outer)

  /* write out hist */
  for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      auto n = __shfl_up_sync(0xffffffff, sum, d);
      if (threadIdx.x % 32 >= d) sum += n;
    }
  }

  for (auto i = 0; i < K; i++)
    if (threadIdx.x % 32 == 31)
      atomicAdd(&s.hist[(int)SmallRadius + i - R], p_hist[i]);
  __syncthreads();

  for (auto i = threadIdx.x; i < 2 * SmallRadius; i += blockDim.x)
    atomicAdd(dram_hist + (i - SmallRadius + radius), s.hist[i]);
  __syncthreads();

  // end of kernel
}

// baseline
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
__global__ void c_lorenzo_3d1l(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;
  __shared__ T s[9][33];
  T delta[TileDim + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * stride3.y + giz_base * stride3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * stride3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < len3.x and giy < len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < len3.z)
          delta[z + 1] =
              round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = ZigZag ? delta : delta + radius;
    if (x < len3.x and y < len3.y and z < len3.z) {
      if (ZigZag)
        eq[gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        eq[gid] = quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = gid;
        cval[cur_idx] = candidate;
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();

  for (auto z = TileDim; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;
    __syncthreads();

    // y-direction, exchange via shmem
    // ghost padding along y
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    // now delta[z] is delta
    quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
  }
}

// add rep to baseline
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
__global__ void c_lorenzo_3d1l_v2(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, dim3 repeat3)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;
  __shared__ T s[9][33];
  T delta[TileDim + 1] = {0};  // first el = 0

  auto gix = [&](auto rx) {
    return blockIdx.x * (TileDim * 4 * repeat3.x) + (TileDim * rx) +
           threadIdx.x;
  };
  auto giy = [&](auto ry) {
    return blockIdx.y * (TileDim * repeat3.y) + (TileDim * ry) + threadIdx.y;
  };
  auto giz_base = [&](auto rz) {
    return blockIdx.z * (TileDim * repeat3.z) + (TileDim * rz);
  };
  auto base_id = [&](auto rx, auto ry, auto rz) {
    return gix(rx) + giy(ry) * stride3.y + giz_base(rz) * stride3.z;
  };

  auto giz = [&](auto rz, auto z) { return giz_base(rz) + z; };
  auto gid = [&](auto rx, auto ry, auto rz, auto z) {
    return base_id(rx, ry, rz) + z * stride3.z;
  };

  auto load_prequant_3d = [&](auto rx, auto ry, auto rz) {
    if (gix(rx) < len3.x and giy(ry) < len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(rz, z) < len3.z)
          delta[z + 1] = round(
              data[gid(rx, ry, rz, z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto _gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = ZigZag ? delta : delta + radius;
    if (x < len3.x and y < len3.y and z < len3.z) {
      if (ZigZag)
        eq[_gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        eq[_gid] = quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = _gid;
        cval[cur_idx] = candidate;
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////
  for (auto rz = 0; rz < repeat3.z; rz++) {
    for (auto ry = 0; ry < repeat3.y; ry++) {
      for (auto rx = 0; rx < repeat3.x; rx++) {
        /////

        // start of the main content
        load_prequant_3d(rx, ry, rz);

        for (auto z = TileDim; z > 0; z--) {
          // z-direction
          delta[z] -= delta[z - 1];

          // x-direction
          auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
          if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;
          __syncthreads();

          // y-direction, exchange via shmem
          // ghost padding along y
          s[threadIdx.y + 1][threadIdx.x] = delta[z];
          __syncthreads();

          delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

          // now delta[z] is delta
          quantize_compact_write(
              delta[z], gix(rx), giy(ry), giz(rz, z - 1),
              gid(rx, ry, rz, z - 1));
        }
        // end of z sequentiality loop
      }
    }
  }
  // end of rep-loop
}

// v3: tune perf on top of v2
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
__global__ void c_lorenzo_3d1l_v3(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, dim3 repeat3)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;
  __shared__ T s[9][33];
  T delta[TileDim + 1] = {0};  // first el = 0

  auto gix = [&]() {  // no repeat on x
    return blockIdx.x * (TileDim * 4) + threadIdx.x;
  };
  auto giy = [&](auto ry) {
    return blockIdx.y * (TileDim * repeat3.y) + (TileDim * ry) + threadIdx.y;
  };
  auto giz_base = [&](auto rz) {
    return blockIdx.z * (TileDim * repeat3.z) + (TileDim * rz);
  };
  auto base_id = [&](auto ry, auto rz) {
    return gix() + giy(ry) * stride3.y + giz_base(rz) * stride3.z;
  };

  auto giz = [&](auto rz, auto z) { return giz_base(rz) + z; };
  auto gid = [&](auto ry, auto rz, auto z) {
    return base_id(ry, rz) + z * stride3.z;
  };

  auto load_prequant_3d = [&](auto ry, auto rz) {
    if (gix() < len3.x and giy(ry) < len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(rz, z) < len3.z)
          delta[z + 1] =
              round(data[gid(ry, rz, z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto _gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = ZigZag ? delta : delta + radius;
    if (x < len3.x and y < len3.y and z < len3.z) {
      if (ZigZag)
        eq[_gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        eq[_gid] = quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = _gid;
        cval[cur_idx] = candidate;
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////
  for (auto rz = 0; rz < repeat3.z; rz++) {
    for (auto ry = 0; ry < repeat3.y; ry++) {
      // start of the main content
      load_prequant_3d(ry, rz);

      for (auto z = TileDim; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;
        __syncthreads();

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_compact_write(
            delta[z], gix(), giy(ry), giz(rz, z - 1), gid(ry, rz, z - 1));
      }
      // end of z sequentiality loop
    }
  }
  // end of rep-loop
}

// v4: add dram hist to v3
template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, int SmallRadius = 128, int TinyRadius = 1>
__global__ void c_lorenzo_3d1l_v4(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn, dim3 repeat3,
    uint32_t* dram_hist)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;
  constexpr auto K = TinyRadius;
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;  // K = 5, R = 2

  __shared__ T s[9][33];
  __shared__ uint32_t s_hist[SmallRadius * 2];

  T delta[TileDim + 1] = {0};  // first el = 0
  uint32_t p_hist[K] = {0};

  auto gix = [&]() {  // no repeat on x
    return blockIdx.x * (TileDim * 4) + threadIdx.x;
  };
  auto giy = [&](auto ry) {
    return blockIdx.y * (TileDim * repeat3.y) + (TileDim * ry) + threadIdx.y;
  };
  auto giz_base = [&](auto rz) {
    return blockIdx.z * (TileDim * repeat3.z) + (TileDim * rz);
  };
  auto base_id = [&](auto ry, auto rz) {
    return gix() + giy(ry) * stride3.y + giz_base(rz) * stride3.z;
  };

  auto giz = [&](auto rz, auto z) { return giz_base(rz) + z; };
  auto gid = [&](auto ry, auto rz, auto z) {
    return base_id(ry, rz) + z * stride3.z;
  };

  auto nworker = [&]() { return blockDim.x * blockDim.y; };

  auto load_prequant_3d = [&](auto ry, auto rz) {
    if (gix() < len3.x and giy(ry) < len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(rz, z) < len3.z)
          delta[z + 1] =
              round(data[gid(ry, rz, z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto _gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = ZigZag ? delta : delta + radius;

    if (x < len3.x and y < len3.y and z < len3.z) {
      if (abs(delta) < SmallRadius) {
        if (2 * abs(delta) < K)
          p_hist[(int)delta + R] += 1;
        else
          atomicAdd(&s_hist[(int)delta + SmallRadius], 1);
      }
      else {
        if (ZigZag)
          atomicAdd(&dram_hist[((EqInt)candidate) * quantizable], 1);
        else
          atomicAdd(&dram_hist[((EqUint)candidate) * quantizable], 1);
      }

      if (ZigZag)
        eq[_gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        eq[_gid] = quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = _gid;
        cval[cur_idx] = candidate;
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////
  /* clear s_hist buffer */

  for (auto i = threadIdx.x + threadIdx.y * blockDim.x; i < 2 * SmallRadius;
       i += nworker())
    s_hist[i] = 0;
  __syncthreads();

  for (auto rz = 0; rz < repeat3.z; rz++) {
    for (auto ry = 0; ry < repeat3.y; ry++) {
      // start of the main content
      load_prequant_3d(ry, rz);

      for (auto z = TileDim; z > 0; z--) {
        // z-direction
        delta[z] -= delta[z - 1];

        // x-direction
        auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
        if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;
        __syncthreads();

        // y-direction, exchange via shmem
        // ghost padding along y
        s[threadIdx.y + 1][threadIdx.x] = delta[z];
        __syncthreads();

        delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

        // now delta[z] is delta
        quantize_compact_write(
            delta[z], gix(), giy(ry), giz(rz, z - 1), gid(ry, rz, z - 1));
      }
      // end of z sequentiality loop
    }
  }
  // end of rep-loop

  /* write out hist */
  for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      auto n = __shfl_up_sync(0xffffffff, sum, d);
      if (threadIdx.x % 32 >= d) sum += n;
    }
  }

  for (auto i = 0; i < K; i++)
    if (threadIdx.x % 32 == 31)
      atomicAdd(&s_hist[(int)SmallRadius + i - R], p_hist[i]);
  __syncthreads();

  for (auto i = threadIdx.x + threadIdx.y * blockDim.x; i < 2 * SmallRadius;
       i += nworker())
    atomicAdd(dram_hist + (i - SmallRadius + radius), s_hist[i]);
  __syncthreads();
}

}  // namespace rolling
}  // namespace psz

#endif /* AAC905A6_6314_4E1E_B5CD_BBBA9005A448 */
