/**
 * @file histsp_serial.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-26
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>

#include "kernel2/histsp.hh"

namespace psz {
namespace detail {

// temporarily, there should be no obvious speed up than the normal hist on
// CPU.
template <typename T, typename FQ>
int histsp_cpu(T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen)
{
  auto radius = outlen / 2;
  T center{0}, neg1{0}, pos1{0};

  for (auto i = 0; i < inlen; i++) {
    auto n = in[i];
    if (n == radius)
      center++;
    else if (n == radius - 1)
      neg1++;
    else if (n == radius + 1)
      pos1++;
    else
      out_hist[n]++;
  }
  out_hist[radius] = center;
  out_hist[radius - 1] = neg1;
  out_hist[radius + 1] = pos1;

  return 0;
}

}  // namespace detail
}  // namespace psz

template <>
int histsp<psz_policy::CPU, uint32_t, uint32_t>(
    uint32_t* in, uint32_t inlen, uint32_t* out_hist, uint32_t outlen,
    void* stream)
{
  return psz::detail::histsp_cpu<uint32_t, uint32_t>(
      in, inlen, out_hist, outlen);
}