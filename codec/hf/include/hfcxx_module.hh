#ifndef ED744237_D186_4707_B7FC_1B931FFC5CBB
#define ED744237_D186_4707_B7FC_1B931FFC5CBB

#include <cstdint>

#include "hf.h"
#include "hfcxx_array.hh"

namespace phf::cuhip {

/**
 * @brief a namespace-like class for batch template instantiations; a rewrite
 * of hfcodec.{hh,cc}; all the included wrapped kernels/methods are `static`
 *
 * @tparam E input type, e.g., error-quantization code in psz
 * @tparam H intermediate type for Huffman coding
 * @tparam TIMING default true to replicate the original
 */
template <typename E, typename H, bool TIMING = true>
class modules {
  // metadata, e.g., saved index for parallel operations
  using M = PHF_METADATA;

 public:
  static void GPU_coarse_encode_phase1(
      hfcxx_array<E> in, hfcxx_array<H> book, const int numSMs,
      hfcxx_array<H> out, float* time_lossless, void* stream);

  static void GPU_coarse_encode_phase1_collect_metadata(
      hfcxx_array<E> in, hfcxx_array<H> book, const int numSMs,
      hfcxx_array<H> out, hfcxx_array<M> par_nbit, hfcxx_array<M> par_ncell,
      hfpar_description hfpar, float* time_lossless, void* stream);

  static void GPU_coarse_encode_phase2(
      hfcxx_array<H> in, hfpar_description hfpar, hfcxx_array<H> deflated,
      hfcxx_array<M> par_nbit, hfcxx_array<M> par_ncell, float* time_lossless,
      void* stream);

  static void GPU_coarse_encode_phase3(
      hfcxx_array<M> d_par_nbit, hfcxx_array<M> d_par_ncell,
      hfcxx_array<M> d_par_entry,  //
      hfpar_description hfpar,     //
      hfcxx_array<M> h_par_nbit, hfcxx_array<M> h_par_ncell,
      hfcxx_array<M> h_par_entry,                 //
      size_t* outlen_nbit, size_t* outlen_ncell,  //
      float* time_cpu_time, void* stream);

  static void GPU_coarse_encode_phase4(
      hfcxx_array<H> buf, hfcxx_array<M> par_entry, hfcxx_array<M> par_ncell,
      hfpar_description hfpar, hfcxx_array<H> bitstream, float* time_lossless,
      void* stream);

  static void GPU_coarse_decode(
      hfcxx_array<H> bitstream, hfcxx_array<uint8_t> revbook,
      hfcxx_array<M> par_nbit, hfcxx_array<M> par_entry,
      hfpar_description hfpar, hfcxx_array<E> out, float* time_lossless,
      void* stream);

  static void GPU_experimental_scatter(
      hfcxx_compact<E> compact, E* out, float* milliseconds, void* stream);
};

}  // namespace phf::cuhip

#endif /* ED744237_D186_4707_B7FC_1B931FFC5CBB */
