#ifndef FREQ_FILTER_CUH
#define FREQ_FILTER_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cuComplex.h>
#include "cufft.h"

#include <stdio.h>
#include<stdlib.h>
#include <vector>

#include "definition.h"
#include "utils.h"

//------------- common

__global__ void symmetrize(cuComplex *inOut, const int n_fft);

__global__ void expandInputForFreqFilter(cuComplex *out, const cuComplex *in, const int inputLen, const int outputLen,
                                         const int n_fft, const int blockLen, const int n_blocks);

__global__ void RemoveInvalidFreqFilterOutput(cuComplex *out, const cuComplex *in, const int outputLen,
                                              const int n_fft, const int blockLen, const int stateLen);

__global__ void combineIQ(cuComplex *out, const float *in_re, const float *in_im,
                          const int dataLen);

__global__ void separateIQ(float *out_re, float *out_im, const cuComplex *in,
                           const int dataLen);

//------------- For Individual filter only, with variable filter coeffs

__global__ void multiply(cuComplex *out, const cuComplex *in, const cuComplex *filterCoeffH_SymbolAddress,
                         const int inputLen, const int n_fft, const int n_blocks);

//------------- For Individual filter only, constant memory array is hard coded

__global__ void multiplyInplace(cuComplex *inOut,
                                const int inputLen, const int n_fft, const int n_blocks);

//------------- For upsample and filter

__global__ void mergeOddEvenIndicesAndSeparateIQ(float *out_re, float *out_im, const cuComplex *in_1,
                                                 const cuComplex *in_2, const int inputLen);

//===================================//

//------------- or Individual filter only, with variable filter coeffs

void initFreqFilterParams(FreqFilterParams& params, cuComplex *filterCoeffsSymbolAddr, const float *filterCoeffs,
                                const int& filterCoeffsLen, const int& inputLen);

void resetFreqFilterParams(FreqFilterParams& params);

void applyFreqDomainFilter(float *out_re, float *out_im, const float *in_re, const float *in_im,
                           FreqFilterParams& params);

//------------- For upsample and filter

void initUpsampleAndFreqFilterParams(UpsampleAndFreqFilterParams& params, const float *filterCoeffs,
                                const int& filterCoeffsLen, const int& inputLen);

void resetUpsampleAndFreqFilterParams(UpsampleAndFreqFilterParams& params);

void applyFreqDomainFilter(float *out_re, float *out_im, float *in_re, float *in_im,
                           UpsampleAndFreqFilterParams& params);

#endif // FREQ_FILTER_CUH
