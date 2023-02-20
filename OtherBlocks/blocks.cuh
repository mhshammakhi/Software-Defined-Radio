#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#define  _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <iostream>

//--- For setting the size of c_bbFilterCoeffs
const int max_bbfilter_len = 1024;
//

//--- Downconverter
__global__
void Baseband(cuComplex inOut[], const float* freq_init,
              const int dataLength, const float frequency);

__global__
void Baseband_Update_State(float* freq_init, const float frequency, const int dataLength);

//--- Baseband Filter
void setBBFilterCoeffsConstMem(const float cpu_array[], const int& filterLen);

__global__
void BasebandFilter(cuComplex output[], const cuComplex input[],
    const int filterLength, const int dataLength);

//--- Demappers
enum class ModulationType {
	BPSK, QPSK, _8PSK, _16QAM
};

int getBitsPerSymbol(const ModulationType& mod_type);

__device__ __forceinline__ void dec2bin(int size, uint8_t z, uint8_t *z_bit);

__global__ void Demapper_BPSK(uint8_t *outputBits, const cuComplex *in, const int dataLen);

__global__ void Demapper_QPSK(uint8_t *outputBits, const cuComplex *in, const int dataLen);

__global__ void Demapper_8PSK(uint8_t *outputBits, const cuComplex *in, const int dataLen);

__global__ void Demapper_16QAM(uint8_t *outputBits, const cuComplex *in, const int dataLen);

void Demapper(const ModulationType& mod_type, uint8_t *outputBits, const cuComplex *in, const int dataLen);