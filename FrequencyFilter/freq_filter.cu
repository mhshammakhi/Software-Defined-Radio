
#include "freq_filter.cuh"
#include<cuComplex.h>

#define DEF_N_FFT 2048
__constant__ cuComplex c_FilterCoeffs_fft[DEF_N_FFT];
__constant__ cuComplex c_FilterCoeffs_Even_fft[DEF_N_FFT];
__constant__ cuComplex c_FilterCoeffs_Odd_fft[DEF_N_FFT];

//------------- common

__global__ void symmetrize(cuComplex *inOut, const int n_fft) {
    int i{ threadIdx.x + blockDim.x * blockIdx.x };
    int middlePointIdx = n_fft / 2;
    while (i < n_fft / 2 - 1) {
        inOut[i + middlePointIdx + 1].x = inOut[middlePointIdx - 1 - i].x;
        inOut[i + middlePointIdx + 1].y = -inOut[middlePointIdx - 1 - i].y;
        i += (blockDim.x * gridDim.x);
    }
}

__global__ void expandInputForFreqFilter(cuComplex *out, const cuComplex *in, const int inputLen, const int outputLen,
                                         const int n_fft, const int blockLen, const int n_blocks) {

    int i{ threadIdx.x + blockDim.x * blockIdx.x };
    int j;
    int idx_in;
    while (i < outputLen) {
        j = i / n_fft;
        idx_in = (i % n_fft + j * blockLen);
        if (idx_in >= inputLen)
            out[i] = { 0., 0. };
        else
            out[i] = in[idx_in];

        i += (blockDim.x * gridDim.x);
    }
}

__global__ void RemoveInvalidFreqFilterOutput(cuComplex *out, const cuComplex *in, const int outputLen,
                                              const int n_fft, const int blockLen, const int stateLen) {

    int i{ threadIdx.x + blockDim.x * blockIdx.x };
    int j;
    while (i < outputLen) {
        j = i / blockLen;
        out[i] = in[i % blockLen + stateLen + j * n_fft];
        i += (blockDim.x * gridDim.x);
    }
}

__global__ void combineIQ(cuComplex *out, const float *in_re, const float *in_im,
                          const int dataLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};

    while(i < dataLen){
        out[i].x = in_re[i];
        out[i].y = in_im[i];
        i += (blockDim.x * gridDim.x);
    }
}

__global__ void separateIQ(float *out_re, float *out_im, const cuComplex *in,
                           const int dataLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};

    while(i < dataLen){
        out_re[i] = in[i].x;
        out_im[i] = in[i].y;
        i += (blockDim.x * gridDim.x);
    }
}

//------------- For Individual filter only but with variable filter coeffs

__global__ void multiply(cuComplex *out, const cuComplex *in, cuComplex *filterCoeffH_SymbolAddress,
                         const int inputLen, const int n_fft, const int n_blocks) {

    int i{ threadIdx.x + blockDim.x * blockIdx.x };
    cuComplex temp;

    while (i < inputLen) {
        temp = in[i];
        out[i].x = (temp.x * filterCoeffH_SymbolAddress[i % n_fft].x - temp.y * filterCoeffH_SymbolAddress[i % n_fft].y)/n_fft;
        out[i].y = (temp.x * filterCoeffH_SymbolAddress[i % n_fft].y + temp.y * filterCoeffH_SymbolAddress[i % n_fft].x)/n_fft;
        i += (blockDim.x * gridDim.x);
    }
}

//------------- For Individual filter only, constant memory array is hard coded

__global__ void multiplyInplace(cuComplex *inOut,
                                const int inputLen, const int n_fft, const int n_blocks) {

    int i{ threadIdx.x + blockDim.x * blockIdx.x };
    cuComplex temp;

    while (i < inputLen) {
        temp = inOut[i];
        inOut[i].x = (temp.x * c_FilterCoeffs_fft[i % n_fft].x - temp.y * c_FilterCoeffs_fft[i % n_fft].y)/n_fft;
        inOut[i].y = (temp.x * c_FilterCoeffs_fft[i % n_fft].y + temp.y * c_FilterCoeffs_fft[i % n_fft].x)/n_fft;
        i += (blockDim.x * gridDim.x);
    }
}

//------------- For upsample and filter

__global__ void mergeOddEvenIndicesAndSeparateIQ(float *out_re, float *out_im, const cuComplex *in_1,
                                                 const cuComplex *in_2, const int inputLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};

    while(i < inputLen){
        out_re[2 * i] = in_1[i].x;
        out_im[2 * i] = in_1[i].y;
        out_re[2 * i + 1] = in_2[i].x;
        out_im[2 * i + 1] = in_2[i].y;
        i += (blockDim.x * gridDim.x);
    }
}

//===================================//

void initFreqFilterParams(FreqFilterParams& params, cuComplex *filterCoeffsSymbolAddr, const float *filterCoeffs,
                                const int& filterCoeffsLen, const int& inputLen){

    params.n_fft = DEF_N_FFT; //if you changed this, change the size of the array in constant memory as well
    params.inputLen = inputLen;

    params.stateLen = filterCoeffsLen - 1;
    params.blockLen = params.n_fft - params.stateLen;
    if((params.inputLen + params.stateLen) % params.blockLen == 0){
        params.n_blocks = (params.inputLen + params.stateLen) / params.blockLen;
        params.has_extra_block = false;
    }
    else{
        params.n_blocks = (params.inputLen + params.stateLen) / params.blockLen + 1;
        params.has_extra_block = true;
    }

    cudaMalloc(&params.d_filterCoeffs, params.n_fft * sizeof(float));
    cudaMemset(params.d_filterCoeffs, 0., params.n_fft * sizeof(float));
    cudaMemcpy(params.d_filterCoeffs, filterCoeffs, filterCoeffsLen * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.d_filterCoeffsFft, params.n_fft * sizeof(cuComplex));
    cufftPlan1d(&params.fftPlanForCoeffs, params.n_fft, CUFFT_R2C, 1);
    cufftExecR2C(params.fftPlanForCoeffs, params.d_filterCoeffs, params.d_filterCoeffsFft);
    cudaFree(params.d_filterCoeffs);

    symmetrize << <params.n_fft/1024, 1024 >> > (params.d_filterCoeffsFft, params.n_fft);
    if(filterCoeffsSymbolAddr == nullptr)
        cudaMemcpyToSymbol(c_FilterCoeffs_fft, params.d_filterCoeffsFft, params.n_fft * sizeof(cuComplex));
    else
        cudaMemcpyAsync(filterCoeffsSymbolAddr, params.d_filterCoeffsFft,
                        params.n_fft * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    cudaMalloc(&params.d_data, (params.inputLen + params.stateLen) * sizeof(cuComplex));
    cudaMalloc(&params.d_data_expanded, params.n_blocks * params.n_fft * sizeof(cuComplex));
    cudaMalloc(&params.d_middle, params.n_blocks * params.n_fft * sizeof(cuComplex));
    cudaMalloc(&params.d_out, params.inputLen * sizeof(cuComplex));
    cudaMalloc(&params.d_out_expanded, params.n_blocks * params.n_fft * sizeof(cuComplex));
    cufftPlan1d(&params.fftPlanForInput, params.n_fft, CUFFT_C2C, params.n_blocks);
    cufftPlan1d(&params.ifftPlanForInput, params.n_fft, CUFFT_C2C, params.n_blocks);

    gpuErrchk();
}

void resetFreqFilterParams(FreqFilterParams& params){
    cudaFree(params.d_filterCoeffsFft);
    cufftDestroy(params.fftPlanForCoeffs);

    cudaFree(params.d_data);
    cudaFree(params.d_data_expanded);
    cudaFree(params.d_middle);
    cudaFree(params.d_out);
    cudaFree(params.d_out_expanded);
    cufftDestroy(params.fftPlanForInput);
    cufftDestroy(params.ifftPlanForInput);

    gpuErrchk();
}

void applyFreqDomainFilter(float *out_re, float *out_im, const float *in_re, const float *in_im,
                           FreqFilterParams& params){

    combineIQ<< <12, 1024 >> >(params.d_data, in_re, in_im, params.inputLen + params.stateLen);

    expandInputForFreqFilter << <12, 1024 >> >(params.d_data_expanded, params.d_data, params.inputLen + params.stateLen,
                                           params.n_fft * params.n_blocks,
                                           params.n_fft, params.blockLen, params.n_blocks);

    cufftExecC2C(params.fftPlanForInput, params.d_data_expanded, params.d_out_expanded, CUFFT_FORWARD);

    multiplyInplace<<<12, 1024>>> (params.d_out_expanded, params.n_fft * params.n_blocks, params.n_fft, params.n_blocks);

    cufftExecC2C(params.ifftPlanForInput, params.d_out_expanded, params.d_out_expanded, CUFFT_INVERSE);

    RemoveInvalidFreqFilterOutput << <12, 1024 >> > (params.d_out, params.d_out_expanded, params.inputLen,
                                                params.n_fft, params.blockLen, params.stateLen);

    separateIQ<< <12, 1024 >> >(out_re, out_im, params.d_out, params.inputLen);

}

//-------------------

void initUpsampleAndFreqFilterParams(UpsampleAndFreqFilterParams& params, const float *filterCoeffs,
                                     const int& filterCoeffsLen, const int& inputLen){

    FreqFilterParams& filt_params = params.freqFilterParams;

    if(!params.upsampleIsActive){
        initFreqFilterParams(filt_params, nullptr, filterCoeffs,
                             filterCoeffsLen, inputLen);
    }
    else{

        std::vector<float> filterCoeffs_Even(filterCoeffsLen/2), filterCoeffs_Odd(filterCoeffsLen/2);
        for(int i{}; i < filterCoeffsLen/2; ++i){
            filterCoeffs_Even[i] = filterCoeffs[2 * i];
            filterCoeffs_Odd[i] = filterCoeffs[2 * i + 1];
        }

        cudaGetSymbolAddress((void **)&filt_params.d_filterCoeffH_SymbolAddress, c_FilterCoeffs_Even_fft);
        initFreqFilterParams(filt_params, filt_params.d_filterCoeffH_SymbolAddress,
                             filterCoeffs_Even.data(), filterCoeffs_Even.size(), inputLen);

        cudaMalloc(&filt_params.d_filterCoeffs, filt_params.n_fft * sizeof(float));
        cudaMemset(filt_params.d_filterCoeffs, 0., filt_params.n_fft * sizeof(float));
        cudaMemcpy(filt_params.d_filterCoeffs, filterCoeffs_Odd.data(), filterCoeffs_Odd.size() * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&params.d_filterCoeffsFft, filt_params.n_fft * sizeof(cuComplex));
        cufftExecR2C(filt_params.fftPlanForCoeffs, filt_params.d_filterCoeffs, params.d_filterCoeffsFft);
        cudaFree(filt_params.d_filterCoeffs);

        symmetrize << <filt_params.n_fft/1024, 1024 >> > (params.d_filterCoeffsFft, filt_params.n_fft);
        cudaGetSymbolAddress((void **)&params.d_filterCoeffH_SymbolAddress, c_FilterCoeffs_Odd_fft);
        cudaMemcpyAsync(params.d_filterCoeffH_SymbolAddress, params.d_filterCoeffsFft,
                        filt_params.n_fft * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

        cudaMalloc(&params.d_out, inputLen * sizeof(cuComplex));
        cudaMalloc(&params.d_out_expanded, filt_params.n_blocks * filt_params.n_fft * sizeof(cuComplex));
    }
}

void resetUpsampleAndFreqFilterParams(UpsampleAndFreqFilterParams& params){

    FreqFilterParams& filt_params = params.freqFilterParams;
    resetFreqFilterParams(filt_params);

    if(params.upsampleIsActive){
        cudaFree(params.d_filterCoeffsFft);
        cudaFree(params.d_out);
        cudaFree(params.d_out_expanded);
    }
}

void applyFreqDomainFilter(float *out_re, float *out_im, float *in_re, float *in_im,
                           UpsampleAndFreqFilterParams& params){

    FreqFilterParams& filt_params = params.freqFilterParams;

    if(!params.upsampleIsActive){
        applyFreqDomainFilter(out_re, out_im, in_re, in_im, filt_params);
    }
    else{
        combineIQ<< <12, 1024 >> >(filt_params.d_data, in_re, in_im,
                                filt_params.inputLen + filt_params.stateLen);

        expandInputForFreqFilter << <12, 1024 >> >(filt_params.d_data_expanded, filt_params.d_data,
                                               filt_params.inputLen + filt_params.stateLen,
                                               filt_params.n_fft * filt_params.n_blocks,
                                               filt_params.n_fft, filt_params.blockLen, filt_params.n_blocks);

        cufftExecC2C(filt_params.fftPlanForInput, filt_params.d_data_expanded,
                     filt_params.d_middle, CUFFT_FORWARD);

        multiply<<<12, 1024>>> (filt_params.d_out_expanded, filt_params.d_middle,
                               filt_params.d_filterCoeffH_SymbolAddress,
                                      filt_params.n_fft * filt_params.n_blocks,
                                      filt_params.n_fft, filt_params.n_blocks);

        cufftExecC2C(filt_params.ifftPlanForInput, filt_params.d_out_expanded,
                     filt_params.d_out_expanded, CUFFT_INVERSE);

        RemoveInvalidFreqFilterOutput << <12, 1024 >> > (filt_params.d_out, filt_params.d_out_expanded,
                                                    filt_params.inputLen,
                                                    filt_params.n_fft, filt_params.blockLen, filt_params.stateLen);

        multiply<<<12, 1024>>> (params.d_out_expanded, filt_params.d_middle,
                               params.d_filterCoeffH_SymbolAddress,
                                      filt_params.n_fft * filt_params.n_blocks,
                                      filt_params.n_fft, filt_params.n_blocks);

        cufftExecC2C(filt_params.ifftPlanForInput, params.d_out_expanded,
                     params.d_out_expanded, CUFFT_INVERSE);

        RemoveInvalidFreqFilterOutput << <12, 1024 >> > (params.d_out, params.d_out_expanded,
                                                    filt_params.inputLen,
                                                    filt_params.n_fft, filt_params.blockLen, filt_params.stateLen);

        mergeOddEvenIndicesAndSeparateIQ<< <12, 1024 >> >(out_re, out_im, filt_params.d_out,
                                         params.d_out, filt_params.inputLen);
    }

    cudaMemcpyAsync(in_re,
                    in_re + filt_params.inputLen,
                    filt_params.stateLen * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaMemcpyAsync(in_im,
                    in_im + filt_params.inputLen,
                    filt_params.stateLen * sizeof(float), cudaMemcpyDeviceToDevice);
}
