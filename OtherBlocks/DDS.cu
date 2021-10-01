__global__
void Baseband(float *d_data_Re, float *d_data_Im,
              float *d_DDC_Re, float *d_DDC_Im,
              const int dataLength, const float frequency)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < dataLength)
    {
        float freq = float(i % 10000) * frequency;
        float omega = 2.0 * pi * (freq - int(freq)) + freq_init;
        //float omega = 0.f;
        d_DDC_Re[i] = d_data_Re[i] * cosf(omega) + d_data_Im[i] * sinf(omega);
        d_DDC_Im[i] = d_data_Im[i] * cosf(omega) - d_data_Re[i] * sinf(omega);
        //i += blockDim.x * gridDim.x;
    }
}
