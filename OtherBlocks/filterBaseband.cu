__global__
void filterBaseband(float *d_data_Re, float *d_data_Im, float *d_filteredData_Re,
                    float *d_filteredData_Im, const int filterLength, const int dataLength, float *d_ABS)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum_re{ 0 }, sum_im{ 0 };
    if (i < dataLength)
    {
        sum_re = 0.0f;
        sum_im = 0.0f;

        for (int j = 0; j < filterLength; j++)
        {

            sum_re += c_baseBandFilter_Coef[j] * d_data_Re[i - j + filterLength - 1];
            sum_im += c_baseBandFilter_Coef[j] * d_data_Im[i - j + filterLength - 1];
        }

        d_filteredData_Re[i] = sum_re;
        d_filteredData_Im[i] = sum_im;
        d_ABS[i] = (sum_re*sum_re + sum_im*sum_im);
        //if(threadIdx.x == 0)
    }
}
