#include "kernels.cuh"

__constant__ float c_baseBandFilter_Coef[5000];

__device__ __managed__ float freq_init{ 0 };

__device__ __managed__ float d_locStep_float{};
__device__ __managed__ int d_locStep_int{};

#define PI 3.14159265359f

__global__
void Baseband(float *d_data_Re, float *d_data_Im,
	float *d_DDC_Re, float *d_DDC_Im,
	const int dataLength, const float frequency)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	while (i < dataLength)
	{
		float freq = float(i % 10000) * frequency;
		float omega = 2.0 * PI * (freq - int(freq)) + freq_init;
		d_DDC_Re[i] = d_data_Re[i] * cosf(omega) + d_data_Im[i] * sinf(omega);
		d_DDC_Im[i] = d_data_Im[i] * cosf(omega) - d_data_Re[i] * sinf(omega);
		i += blockDim.x * gridDim.x;
	}
}

__global__
void SetFreq_init(const int SOF, float freq)
{
	float freqLast = float(SOF % 10000) * freq;
	freq_init += 2.0*PI*(freqLast - int(freqLast));
}

__global__
void BasebandFilter(float *d_data_Re, float *d_data_Im, float *d_filteredData_Re,
	float *d_filteredData_Im, const int filterLength, const int dataLength)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float sum_re{ 0 }, sum_im{ 0 };
	while (i < dataLength)
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

		i += (blockDim.x * gridDim.x);
	}
}

void separateIQ_CPU(float *data_re, float *data_im, const cuComplex *data_comp, const int& dataLen) {
	for (int i{}; i < dataLen; ++i) {
		data_re[i] = data_comp[i].x;
		data_im[i] = data_comp[i].y;
	}
}

void combineIQ_CPU(cuComplex *data_comp, const float *data_re, const float *data_im, const int& dataLen) {
	for (int i{}; i < dataLen; ++i) {
		data_comp[i].x = data_re[i];
		data_comp[i].y = data_im[i];
	}
}

void SDR_Implement(SDR_Params& sdr_params, const SignalParams& sig_params) {
	
	cudaMemcpyToSymbol(c_baseBandFilter_Coef, sdr_params.filterbb_coeffs.data(), sdr_params.filterbb_len * sizeof(float));

	gpuErrchk();

	PartialFileReader fileReader;
	fileReader.setFileName(sig_params.inputFileAddress);
	fileReader.openFile();
	uint32_t num_elements = fileReader.getTotalFileSizeInBytes() / (2 * sizeof(float));

	PartialFileWriter fileWriter;
	fileWriter.setFileName(sig_params.outputFileAddress);
	fileWriter.openFile();

	int outLen_resample{};
	const int n_threads_resample{ 512 }, n_blocks_resample{ 10 };
	int i{};
	while ((i + 1) * sdr_params.sizeOfFrame <= num_elements) {

		std::cout << "------------------ frame " << i + 1 << " -----------------\n";

		fileReader.readBinData(sdr_params.input, sdr_params.sizeOfFrame);
		separateIQ_CPU(sdr_params.input_re, sdr_params.input_im, sdr_params.input, sdr_params.sizeOfFrame);
		cudaMemcpyAsync(sdr_params.d_ddcIn_re, sdr_params.input_re, sizeof(float) * sdr_params.sizeOfFrame, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(sdr_params.d_ddcIn_im, sdr_params.input_im, sizeof(float) * sdr_params.sizeOfFrame, cudaMemcpyHostToDevice);

		//gpuErrchk();

		Baseband << < 12, 1024 >> >(sdr_params.d_ddcIn_re,
			sdr_params.d_ddcIn_im,
			sdr_params.d_ddcOut_re + sdr_params.filterbb_len - 1,
			sdr_params.d_ddcOut_im + sdr_params.filterbb_len - 1,
			sdr_params.sizeOfFrame,
			sig_params.frequency);

		SetFreq_init << <1, 1 >> >(sdr_params.sizeOfFrame, sig_params.frequency);

		//gpuErrchk();

		
		applyFreqDomainFilter(sdr_params.d_bbFilterOut_re + 4, sdr_params.d_bbFilterOut_im + 4,
			sdr_params.d_ddcOut_re, sdr_params.d_ddcOut_im,
			sdr_params.upsampleAndFreqFilterParams);

		//gpuErrchk();
		
		std::cout << "filterLen: " << sdr_params.sizeOfFrame << std::endl;
		cudaMemcpy(sdr_params.h_out_re, sdr_params.d_bbFilterOut_re + 4, sizeof(float) * sdr_params.sizeOfFrame, cudaMemcpyDeviceToHost);
		cudaMemcpy(sdr_params.h_out_im, sdr_params.d_bbFilterOut_im + 4, sizeof(float) * sdr_params.sizeOfFrame, cudaMemcpyDeviceToHost);

		combineIQ_CPU(sdr_params.output, sdr_params.h_out_re, sdr_params.h_out_im, sdr_params.sizeOfFrame);
		fileWriter.writeBinData(sdr_params.output, sdr_params.sizeOfFrame);

		gpuErrchk();
		i++;
	}
}