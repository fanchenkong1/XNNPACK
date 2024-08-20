// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c8-wasmdot.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c8__wasmusdot_u2(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x01 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc0x23 = wasm_u64x2_load32x2(w); w = (const int32_t*) w + 2;
    v128_t vacc1x01 = vacc0x01;
    v128_t vacc2x01 = vacc0x01;
    v128_t vacc3x01 = vacc0x01;
    v128_t vacc1x23 = vacc0x23;
    v128_t vacc2x23 = vacc0x23;
    v128_t vacc3x23 = vacc0x23;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        const v128_t va0x0123 = wasm_v128_xor(wasm_v128_load64_splat(a0), vsign_mask);
        const v128_t va0x4567 = wasm_v128_xor(wasm_v128_load64_splat((const int8_t*) a0 + 8), vsign_mask);
        a0 += 16;
        const v128_t va1x0123 = wasm_v128_xor(wasm_v128_load64_splat(a1), vsign_mask);
        const v128_t va1x4567 = wasm_v128_xor(wasm_v128_load64_splat((const int8_t*) a1 + 8), vsign_mask);
        a1 += 16;
        const v128_t va2x0123 = wasm_v128_xor(wasm_v128_load64_splat(a2), vsign_mask);
        const v128_t va2x4567 = wasm_v128_xor(wasm_v128_load64_splat((const int8_t*) a2 + 8), vsign_mask);
        a2 += 16;
        const v128_t va3x0123 = wasm_v128_xor(wasm_v128_load64_splat(a3), vsign_mask);
        const v128_t va3x4567 = wasm_v128_xor(wasm_v128_load64_splat((const int8_t*) a3 + 8), vsign_mask);
        a3 += 16;

        const v128_t vb01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
        const v128_t vb23 = wasm_v128_load(w); w = (const int8_t*) w + 16;
        const v128_t vb01x4567 = wasm_v128_load(w); w = (const int8_t*) w + 16;
        const v128_t vb23x4567 = wasm_v128_load(w); w = (const int8_t*) w + 16;

        vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va0x0123, vacc0x01);
        vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va0x0123, vacc0x23);
        vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va1x0123, vacc1x01);
        vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va1x0123, vacc1x23);
        vacc2x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va2x0123, vacc2x01);
        vacc2x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va2x0123, vacc2x23);
        vacc3x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va3x0123, vacc3x01);
        vacc3x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va3x0123, vacc3x23);
        vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x4567, va0x4567, vacc0x01);
        vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x4567, va0x4567, vacc0x23);
        vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x4567, va1x4567, vacc1x01);
        vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x4567, va1x4567, vacc1x23);
        vacc2x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x4567, va2x4567, vacc2x01);
        vacc2x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x4567, va2x4567, vacc2x23);
        vacc3x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01x4567, va3x4567, vacc3x01);
        vacc3x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23x4567, va3x4567, vacc3x23);
        
        k -= 16 * sizeof(int8_t);
      }

      if (k != 0) {
        const v128_t va0x0123 = wasm_v128_xor(wasm_v128_load64_splat(a0), vsign_mask);
        a0 += 8;
        const v128_t va1x0123 = wasm_v128_xor(wasm_v128_load64_splat(a1), vsign_mask);
        a1 += 8;
        const v128_t va2x0123 = wasm_v128_xor(wasm_v128_load64_splat(a2), vsign_mask);
        a2 += 8;
        const v128_t va3x0123 = wasm_v128_xor(wasm_v128_load64_splat(a3), vsign_mask);
        a3 += 8;

        const v128_t vb01 = wasm_v128_load(w); w = (const int8_t*) w + 16;
        const v128_t vb23 = wasm_v128_load(w); w = (const int8_t*) w + 16;

        vacc0x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va0x0123, vacc0x01);
        vacc0x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va0x0123, vacc0x23);
        vacc1x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va1x0123, vacc1x01);
        vacc1x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va1x0123, vacc1x23);
        vacc2x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va2x0123, vacc2x01);
        vacc2x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va2x0123, vacc2x23);
        vacc3x01 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb01, va3x0123, vacc3x01);
        vacc3x23 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb23, va3x0123, vacc3x23);
        
        k -= 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x01, vacc0x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc0x01, vacc0x23, 1, 3, 5, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x01, vacc1x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc1x01, vacc1x23, 1, 3, 5, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x01, vacc2x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc2x01, vacc2x23, 1, 3, 5, 7));
    v128_t vacc3x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x01, vacc3x23, 0, 2, 4, 6), wasm_v32x4_shuffle(vacc3x01, vacc3x23, 1, 3, 5, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vmagic_bias);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vmagic_min);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc2x0123 = wasm_i32x4_sub(vacc2x0123, vmagic_bias_less_output_zero_point);
    vacc3x0123 = wasm_i32x4_sub(vacc3x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);
    v128_t vacc23x0123 = wasm_i16x8_narrow_i32x4(vacc2x0123, vacc3x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc23x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c3, vout, 3);
      wasm_v128_store32_lane(c2, vout, 2);
      wasm_v128_store32_lane(c1, vout, 1);
      wasm_v128_store32_lane(c0, vout, 0);

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c3, vout, 6);
        c3 += 2;
        wasm_v128_store16_lane(c2, vout, 4);
        c2 += 2;
        wasm_v128_store16_lane(c1, vout, 2);
        c1 += 2;
        wasm_v128_store16_lane(c0, vout, 0);
        c0 += 2;

        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c3, vout, 12);
        wasm_v128_store8_lane(c2, vout, 8);
        wasm_v128_store8_lane(c1, vout, 4);
        wasm_v128_store8_lane(c0, vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
