
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>

#include "blocks.cuh"
#include "../utils.h"

void test_baseband() {

    const std::string signalFileAddress = "input.bin";
    const int frameLen = 2 * 1024 * 1024;
    const float frequency = 0.1f;

    PartialFileReader fileReader;
    fileReader.setFileName(signalFileAddress);
    fileReader.openFile();
    int num_elements = fileReader.getTotalFileSizeInBytes() / sizeof(cuComplex);
    
    PartialFileWriter fileWriter;
    fileWriter.setFileName("output.bin");
    fileWriter.openFile();

    std::vector<cuComplex> h_inOut(frameLen);
    cuComplex* d_inOut;
    cudaMalloc(&d_inOut, frameLen * sizeof(cuComplex));
    float* d_freq_init;
    cudaMalloc(&d_freq_init, sizeof(float));

    int i{};
    while ((i + 1) * frameLen <= num_elements) {
        fileReader.readBinData(h_inOut, frameLen);
        cudaMemcpyAsync(d_inOut, h_inOut.data(), frameLen * sizeof(cuComplex), cudaMemcpyHostToDevice);

        Baseband << <12, 1024 >> > (d_inOut, d_freq_init, frameLen, frequency);
        Baseband_Update_State<<<1, 1>>>(d_freq_init, frequency, frameLen);

        cudaMemcpy(h_inOut.data(), d_inOut, frameLen * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        fileWriter.writeBinData(h_inOut, frameLen);
        i++;
    }

    fileReader.closeFile();
    fileWriter.closeFile();

    cudaFree(d_inOut);
    cudaFree(d_freq_init);
    gpuErrchk();
}

void test_bbfilter() {

    const std::string signalFileAddress = "input.bin";
    const std::string filterCoeffsFileAddress = "filter_coeffs.bin";
    const int frameLen = 2 * 1024 * 1024;

    PartialFileReader fileReader;
    fileReader.setFileName(signalFileAddress);
    fileReader.openFile();
    int num_elements = fileReader.getTotalFileSizeInBytes() / sizeof(cuComplex);

    PartialFileWriter fileWriter;
    fileWriter.setFileName("output.bin");
    fileWriter.openFile();

    std::vector<float> h_filterCoeffs;
    readBinData(h_filterCoeffs, filterCoeffsFileAddress);
    const int bbfilterLen = h_filterCoeffs.size();
    setBBFilterCoeffsConstMem(h_filterCoeffs.data(), h_filterCoeffs.size());

    std::vector<cuComplex> h_inOut(frameLen);
    cuComplex* d_input, *d_output;
    cudaMalloc(&d_input, (frameLen + bbfilterLen - 1) * sizeof(cuComplex));
    cudaMemsetAsync(d_input, 0.f, (bbfilterLen - 1) * sizeof(cuComplex));
    cudaMalloc(&d_output, frameLen * sizeof(cuComplex));
    gpuErrchk();

    int i{};
    while ((i + 1) * frameLen <= num_elements) {
        fileReader.readBinData(h_inOut, frameLen);
        cudaMemcpyAsync(d_input + bbfilterLen - 1, h_inOut.data(), frameLen * sizeof(cuComplex), cudaMemcpyHostToDevice);

        BasebandFilter << <12, 1024 >> > (d_output, d_input, bbfilterLen, frameLen);
        cudaMemcpyAsync(d_input, d_input + frameLen, (bbfilterLen - 1) * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

        cudaMemcpy(h_inOut.data(), d_output, frameLen * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        fileWriter.writeBinData(h_inOut, frameLen);
        i++;
    }

    fileReader.closeFile();
    fileWriter.closeFile();

    cudaFree(d_input);
    cudaFree(d_output);
    gpuErrchk();
}

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    //test_baseband();
    test_bbfilter();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
