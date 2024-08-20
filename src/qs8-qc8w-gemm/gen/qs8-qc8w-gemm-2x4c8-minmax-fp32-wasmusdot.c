// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-wasmdot.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmusdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }
  
  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x01 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc0x23 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc1x01 = vacc0x01;
    v128_t vacc1x23 = vacc0x23;

    size_t k = kc;
    while (k != 0) {
      const v128_t va0x0123 = wasm_v128_xor(wasm_v128_load64_splat(a0), vsign_mask);
      a0 += 8;
      const v128_t va1x0123 = wasm_v128_xor(wasm_v128_load64_splat(a1), vsign_mask);
      a1 += 8;

      const v128_t vb01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb23 = wasm_v128_load(w); w = (const int8_t*) w + 16;

      vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va0x0123, vacc0x01);
      vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va0x0123, vacc0x23);
      vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va1x0123, vacc1x01);
      vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va1x0123, vacc1x23);
      
      k -= 8 * sizeof(int8_t);
    }


    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x01, vacc0x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc0x01, vacc0x23, 1, 3, 5, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x01, vacc1x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc1x01, vacc1x23, 1, 3, 5, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);

    v128_t vacc = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc01x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vacc = wasm_i8x16_min(vacc, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store32_lane(c0, vacc, 0);
      wasm_v128_store32_lane(c1, vacc, 1);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc, 0);
        c0 += 2;
        wasm_v128_store16_lane(c1, vacc, 2);
        c1 += 2;

        vacc = wasm_u32x4_shr(vacc, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc, 0);
        wasm_v128_store8_lane(c1, vacc, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}
