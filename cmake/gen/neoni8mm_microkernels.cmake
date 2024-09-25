# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for neoni8mm
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_NEONI8MM_MICROKERNEL_SRCS
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c8-minmax-neoni8mm.c
  src/qp8-f32-qb4w-gemm/qp8-f32-qb4w-gemm-minmax-8x4c16s2-mstep2-neoni8mm.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-8x8c16s2-mstep2-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c8-minmax-fp32-neoni8mm.c)

SET(NON_PROD_NEONI8MM_MICROKERNEL_SRCS
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x32c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x32c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x32c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x32c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x32-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x32-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x32-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x32-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x32-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x32-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x16c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-neoni8mm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-neoni8mm.c
  src/qp8-f32-qb4w-gemm/qp8-f32-qb4w-gemm-minmax-4x8c16s2-neoni8mm.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-4x4c16s2-neoni8mm.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-4x8c16s2-neoni8mm.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-8x4c16s2-mstep2-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-neoni8mm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-fp32-neoni8mm.c)

SET(ALL_NEONI8MM_MICROKERNEL_SRCS ${PROD_NEONI8MM_MICROKERNEL_SRCS} + ${NON_PROD_NEONI8MM_MICROKERNEL_SRCS})
