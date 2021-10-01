__global__
void filterAndDownSample(float *d_data_Re, float *d_data_Im,
                         float *d_filteredData_Re,float *d_filteredData_Im,
                         const int filterLength, const int dataLength,
                         const int decimationFactor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum_Re;
    float sum_Im;
    if (i < dataLength)
    {
        sum_Re = 0.0;
        sum_Im = 0.0;

        for (int j = 0; j < filterLength; j++)
        {
            sum_Re += c_decimationFilter_Coef[j] * d_data_Re[i*decimationFactor - j+filterLength-1];
            sum_Im += c_decimationFilter_Coef[j] * d_data_Im[i*decimationFactor - j+filterLength-1];
        }

        d_filteredData_Re[i] = sum_Re;
        d_filteredData_Im[i] = sum_Im;
    }
}