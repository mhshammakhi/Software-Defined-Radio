
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "tools.cuh"

//#define _BUILD_LIBS

inline void readBinData(std::vector<float>& re, std::vector<float>& im, std::string fileName, const int dataLen = 0)
{
	std::ifstream signalFile;
	signalFile.open(fileName, std::ios::binary);

	if (!signalFile)
	{
		std::cout << " Error, Couldn't find the input file" << "\n";
		exit(0);
	}
	int num_elements{};
	signalFile.seekg(0, std::ios::end);
	num_elements = signalFile.tellg() / sizeof(float) / 2;
	signalFile.seekg(0, std::ios::beg);
	std::cout << "number of samples in the file: " << num_elements << std::endl;
	if (dataLen != 0 && dataLen <= num_elements)
		num_elements = dataLen;

	re.resize(num_elements);
	im.resize(num_elements);
	std::string signalLine;
	int i{};
	float f;

	std::cout << "first 10 inputs: " << std::endl;
	while ((!signalFile.eof()) && (i < num_elements))
	{
		signalFile.read(reinterpret_cast<char*>(&f), sizeof(float));
		re[i] = f;
		signalFile.read(reinterpret_cast<char*>(&f), sizeof(float));
		im[i] = f;
		if (i < 10)
			std::cout << i << ": " << re[i] << ", " << im[i] << std::endl;
		i++;
	}
	std::cout << "number of read samples: " << i << std::endl;
	signalFile.close();
}

inline void recordData(float *data_re, float *data_im, int sizeOfWrite, std::string fileName)
{
	std::ofstream outFile;
	outFile.open(fileName, std::ios::binary);
	if (outFile.is_open()) {
		std::cout << "isOpen, writing " << sizeOfWrite << " symbols to .bin file" << std::endl;
	}
	else
		std::cout << "isNotOpen" << std::endl;

	for (int i = 0; i < sizeOfWrite; i++)
	{
		outFile.write(reinterpret_cast<const char*>(&data_re[i]), sizeof(float));
		outFile.write(reinterpret_cast<const char*>(&data_im[i]), sizeof(float));
	}
	outFile.close();
	std::cout << "ok" << std::endl;
}

class CudaTimer {
public:
	CudaTimer() {
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
	}
	void start() { cudaEventRecord(_start, 0); }
	void stop() { cudaEventRecord(_stop, 0); }
	float elapsed() { /// return in miliseconds
		cudaEventSynchronize(_stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, _start, _stop);
		return elapsedTime;
	}

private:
	cudaEvent_t _start, _stop;
};

#define gpuErrchk() { gpuAssert(__FILE__, __LINE__); }
inline void gpuAssert(const char *file, int line, bool abort = true)
{
	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int main()
{
#ifndef _BUILD_LIBS
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	//config
	const int sizeOfFrame = 80'000;
	const int numFrames = 96;
	std::string signalAddress{ "E:\\Projects_Part2\\TimingRecovery_PLL\\ForCuda\\pll_input.bin" };
	//

	const float pll_alpha{ 10 }, pll_beta{ 20 };
	const int pow{ 1 };
	std::vector<float> data_re, data_im;
	readBinData(data_re, data_im, signalAddress, sizeOfFrame * numFrames);
	int num_elements = data_re.size();

	if (sizeOfFrame * numFrames > num_elements) {
		std::cout << "Not enough input symbols; reduce the number of frames and try again.\n";
		exit(0);
	}
	std::cout << "inputLen = " << num_elements << ", numFrames = " << numFrames << ", sizeOfFrame = " << sizeOfFrame << std::endl;

	float *d_data_re, *d_data_im;
	cudaMalloc(&d_data_re, sizeOfFrame * numFrames * sizeof(float));
	cudaMalloc(&d_data_im, sizeOfFrame * numFrames * sizeof(float));
	cudaMemcpyAsync(d_data_re, data_re.data(), sizeOfFrame * numFrames * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_data_im, data_im.data(), sizeOfFrame * numFrames * sizeof(float), cudaMemcpyHostToDevice);
	data_re.resize(0);
	data_im.resize(0);

	float *d_PLL_out_re, *d_PLL_out_im;
	cudaMalloc(&d_PLL_out_re, sizeOfFrame * numFrames * sizeof(float));
	cudaMalloc(&d_PLL_out_im, sizeOfFrame * numFrames * sizeof(float));

	float *d_phaseMismatch_first, *d_phaseMismatch_last;
	cudaMalloc(&d_phaseMismatch_first, numFrames * sizeof(float));
	cudaMalloc(&d_phaseMismatch_last, numFrames * sizeof(float));

	float *d_misang;
	cudaMalloc(&d_misang, numFrames * sizeof(float));

	gpuErrchk(cudaGetLastError());

	std::vector<float> pll_out_re(sizeOfFrame * numFrames), pll_out_im(sizeOfFrame * numFrames);

	CudaTimer timer;
	timer.start();

	int repeat{ 100 }; //increase for time measurement
	for (int i{}; i < repeat; i++) {
		PLL_P1 << <numFrames, 1 >> > (d_PLL_out_re, d_PLL_out_im, d_phaseMismatch_first, d_phaseMismatch_last,
			d_data_re, d_data_im, pll_alpha, pll_beta, pow, sizeOfFrame, numFrames);

		mismatchDetection_V2_P1 << < numFrames, 1 >> > (d_misang, d_phaseMismatch_first, d_phaseMismatch_last, numFrames);

		mismatchDetection_V2_P2 << < 1, 1 >> > (d_misang, numFrames);

		PLL_P2 << <12, 1024 >> > (d_PLL_out_re, d_PLL_out_im, d_misang, sizeOfFrame, numFrames);
	}

	timer.stop();
	float elapsed = timer.elapsed();
	std::cout << "Average elapsed time: " << elapsed / repeat << " ms." << std::endl;

	cudaMemcpy(pll_out_re.data(), d_PLL_out_re, sizeOfFrame * numFrames * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(pll_out_im.data(), d_PLL_out_im, sizeOfFrame * numFrames * sizeof(float), cudaMemcpyDeviceToHost);
	recordData(pll_out_re.data(), pll_out_im.data(), sizeOfFrame * numFrames, "pll_out_gpu.bin");

	gpuErrchk(cudaGetLastError());

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	std::cout << "success\n";
#endif
	return 0;
}
