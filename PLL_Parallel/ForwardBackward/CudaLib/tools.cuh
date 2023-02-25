#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <math.h>

typedef struct complex8 { float re; float im; };

__global__ void PLL_P1(float *out_re, float *out_im, float *phiBegin, float *phiEnd,
	const float *in_re, const float *in_im, const float alpha, const float beta, const int pow,
	const int sizeOfFrame, const int numFrames);

__global__ void mismatchDetection_V2_P1(float *misang, const float *phiBegin, const float *phiEnd,
	const int numFrames);

__global__ void mismatchDetection_V2_P2(float *misang, const int numFrames);

__global__ void PLL_P2(float *in_out_re, float *in_out_im, const float *misang,
	const int sizeOfFrame, const int numFrames);