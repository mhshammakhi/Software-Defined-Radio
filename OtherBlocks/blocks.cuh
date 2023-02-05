#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#define  _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

const int max_bbfilter_len = 1024;

__global__
void Baseband(cuComplex inOut[], const float* freq_init,
              const int dataLength, const float frequency);

__global__
void Baseband_Update_State(float* freq_init, const float frequency, const int dataLength);

void setBBFilterCoeffsConstMem(const float cpu_array[], const int& filterLen);

__global__
void BasebandFilter(cuComplex output[], const cuComplex input[],
    const int filterLength, const int dataLength);
