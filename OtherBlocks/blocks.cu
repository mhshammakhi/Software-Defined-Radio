#include "blocks.cuh"

__constant__ float c_bbFilterCoeffs[max_bbfilter_len];

__global__
void Baseband(cuComplex inOut[], const float* freq_init,
    const int dataLength, const float frequency)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    float local_freq_init = *freq_init;
    cuComplex local_input;

    float freq, omega;
    while (i < dataLength)
    {
        float freq = float(i % 10000) * frequency;
        omega = 2.0 * M_PI * (freq - int(freq)) + local_freq_init;
        local_input = inOut[i];
        inOut[i].x = local_input.x * cosf(omega) + local_input.y * sinf(omega);
        inOut[i].y = local_input.y * cosf(omega) - local_input.x * sinf(omega);
        i += blockDim.x * gridDim.x;
    }
}

__global__
void Baseband_Update_State(float* freq_init, const float frequency, const int dataLength) {
    float freqLast = float(dataLength % 10000) * frequency;
    *freq_init += 2.0 * M_PI * (freqLast - int(freqLast));
}

void setBBFilterCoeffsConstMem(const float cpu_array[], const int& filterLen) {
    assert(max_bbfilter_len >= filterLen);
    cudaMemcpyToSymbolAsync(c_bbFilterCoeffs, cpu_array, filterLen * sizeof(float), 0, cudaMemcpyHostToDevice);
}

__global__
void BasebandFilter(cuComplex output[], const cuComplex input[],
    const int filterLength, const int dataLength) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    cuComplex sum;
    while (i < dataLength)
    {
        sum = { 0.f, 0.f };

        for (int j = 0; j < filterLength; j++)
        {
            sum.x += c_bbFilterCoeffs[j] * input[i - j + filterLength - 1].x;
            sum.y += c_bbFilterCoeffs[j] * input[i - j + filterLength - 1].y;
        }

        output[i] = sum;
        i += (blockDim.x * gridDim.x);
    }
}