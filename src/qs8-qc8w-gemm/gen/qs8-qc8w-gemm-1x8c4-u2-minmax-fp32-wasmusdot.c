// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-wasmdot.c.in
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

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__wasmusdot_u2(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x0123 = wasm_v128_load(w); w = (const int32_t*) w + 4;
    v128_t vacc0x4567 = wasm_v128_load(w); w = (const int32_t*) w + 4;
    v128_t vacc1x0x0123 = wasm_i32x4_const(0,0,0,0);
    v128_t vacc1x0x4567 = wasm_i32x4_const(0,0,0,0);

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const v128_t va0x0123 = wasm_v128_xor(wasm_v128_load32_splat(a0), vsign_mask);
      const v128_t va0x4567 = wasm_v128_xor(wasm_v128_load32_splat((const int8_t*) a0 + 4), vsign_mask);
      a0 += 8;

      const v128_t vb0123 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb4567 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb0123x4567 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb4567x4567 = wasm_v128_load(w); w = (const int8_t*) w + 16;

      vacc0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va0x0123, vacc0x0123);
      vacc0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va0x0123, vacc0x4567);
      vacc1x0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123x4567, va0x4567, vacc1x0x0123);
      vacc1x0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567x4567, va0x4567, vacc1x0x4567);

      k -= 8 * sizeof(int8_t);
    }

    vacc0x0123 = wasm_i32x4_add(vacc0x0123, vacc1x0x0123);
    vacc0x4567 = wasm_i32x4_add(vacc0x4567, vacc1x0x4567);
    
    if (k != 0) {
      const v128_t va0x0123 = wasm_v128_xor(wasm_v128_load32_splat(a0), vsign_mask);
      a0 += 4;

      const v128_t vb0123 = wasm_v128_load(w); w = (const int8_t*) w + 16;
      const v128_t vb4567 = wasm_v128_load(w); w = (const int8_t*) w + 16;

      vacc0x0123 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0123, va0x0123, vacc0x0123);
      vacc0x4567 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb4567, va0x0123, vacc0x4567);

      k -= 4 * sizeof(int8_t);
    }

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc0x4567 = wasm_f32x4_convert_i32x4(vacc0x4567);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    const v128_t vscale4567 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc0x4567 = wasm_f32x4_add(vacc0x4567, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc0x4567 = wasm_i32x4_sub(vacc0x4567, vmagic_bias_less_output_zero_point);

    v128_t vacc0x01234567 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x4567);

    v128_t vacc0x01234567_0x01234567 = wasm_i8x16_narrow_i16x8(vacc0x01234567, vacc0x01234567);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vacc0x01234567_0x01234567 = wasm_i8x16_min(vacc0x01234567_0x01234567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store64_lane(c0, vacc0x01234567_0x01234567, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store32_lane(c0, vacc0x01234567_0x01234567, 0);
        c0 += 4;

        vacc0x01234567_0x01234567 = wasm_u64x2_shr(vacc0x01234567_0x01234567, 32);
      }
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc0x01234567_0x01234567, 0);
        c0 += 2;

        vacc0x01234567_0x01234567 = wasm_u32x4_shr(vacc0x01234567_0x01234567, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc0x01234567_0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
