#include "kernels.cuh"

__constant__ float c_baseBandFilter_Coef[5000];
__constant__ float c_matchFilter_Coef[129];

__device__ __managed__ float freq_init{ 0 };
__device__ __managed__ float d_NormalizeFactor_MF{ 1 };

__device__ __managed__ float d_resample_prev_re[4]{}, d_resample_prev_im[4]{};
__device__ __managed__ int d_resample_nFromPrev{};
__device__ __managed__ float d_last_startInterpIndex{ 5 };
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

void resampler_calc_step_host(const int& num_threads, const int& num_blocks, const float& iFs, const float& oFs) {
	double decimationRate{ (double)iFs / (double)oFs };
	double location_step = num_threads * num_blocks * decimationRate;
	d_locStep_int = floor(location_step);
	d_locStep_float = (float)(location_step - floor(location_step));
}

__device__ __forceinline__ void resampler_cubic_interp(const float *in_re, const float *in_im, const float& mu, float *out_re, float *out_im) {
	float v_re[4], v_im[4];
	v_re[0] = in_re[1];
	v_re[1] = -1.f / 6 * in_re[3] + in_re[2] - 1.f / 2 * in_re[1] - 1.f / 3 * in_re[0];
	v_re[2] = 1.f / 2 * in_re[2] - in_re[1] + 1.f / 2 * in_re[0];
	v_re[3] = 1.f / 6 * in_re[3] - 1.f / 2 * in_re[2] + 1.f / 2 * in_re[1] - 1.f / 6 * in_re[0];

	v_im[0] = in_im[1];
	v_im[1] = -1.f / 6 * in_im[3] + in_im[2] - 1.f / 2 * in_im[1] - 1.f / 3 * in_im[0];
	v_im[2] = 1.f / 2 * in_im[2] - in_im[1] + 1.f / 2 * in_im[0];
	v_im[3] = 1.f / 6 * in_im[3] - 1.f / 2 * in_im[2] + 1.f / 2 * in_im[1] - 1.f / 6 * in_im[0];

	out_re[0] = ((v_re[3] * mu + v_re[2]) * mu + v_re[1]) * mu + v_re[0];
	out_im[0] = ((v_im[3] * mu + v_im[2]) * mu + v_im[1]) * mu + v_im[0];
}

__global__ void resampler(float *out_re, float *out_im, int *outLen, float *in_re, float *in_im,
	const float iFs, const float oFs, const int frameLen) {
	float decimationRate{ iFs / oFs };
	int startInterpInputIndex{ 1 - d_resample_nFromPrev };
	float startInterpIndex{ d_last_startInterpIndex };
	float outlen_f{ ((frameLen + 4) - startInterpIndex - 2) / decimationRate };
	int outlen;
	if (outlen_f - floorf(outlen_f) > 0)
		outlen = floorf(outlen_f) + 1;
	else
		outlen = floorf(outlen_f);

	int i{ threadIdx.x + blockIdx.x * blockDim.x };
	float location_float{ (i * decimationRate + startInterpIndex) - floorf(i * decimationRate + startInterpIndex) }, locStep_float{ d_locStep_float };
	int interpInputIndex{ static_cast<int>(i * decimationRate + startInterpIndex) }, locStep_int{ d_locStep_int };

	int resample_nFromPrev_local{ d_resample_nFromPrev };
	float interpIndexPos{}, interpClockPhase{};

	//if blockDim.x < 4 take this to the while loop
	if (i < 4) {
		in_re[i] = d_resample_prev_re[i];
		in_im[i] = d_resample_prev_im[i];
	}
	__syncthreads();

	while (i < outlen) {
		resampler_cubic_interp(in_re + interpInputIndex - 1, in_im + interpInputIndex - 1, location_float, out_re + i, out_im + i);

		if (i == (outlen - 1))
		{
			d_last_startInterpIndex = decimationRate + interpInputIndex + location_float - frameLen;

			for (int tmp{ 0 }; tmp < 4; tmp++) {
				d_resample_prev_re[tmp] = in_re[frameLen + tmp];
				d_resample_prev_im[tmp] = in_im[frameLen + tmp];
			}
			outLen[0] = outlen;
		}

		location_float += locStep_float;
		interpInputIndex += locStep_int + floorf(location_float);
		location_float -= floorf(location_float);

		i += blockDim.x * gridDim.x;
	}
}

__global__
void MatchFilter(float *d_data_Re, float *d_data_Im, float *d_filteredData_Re,
	float *d_filteredData_Im, float *d_ABS, const int filterLength, const int dataLength)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float sum_re{ 0 }, sum_im{ 0 };
	while (i < dataLength)
	{
		sum_re = 0.0f;
		sum_im = 0.0f;

		for (int j = 0; j < filterLength; j++)
		{
			sum_re += c_matchFilter_Coef[j] * d_data_Re[i - j + filterLength - 1];
			sum_im += c_matchFilter_Coef[j] * d_data_Im[i - j + filterLength - 1];
		}

		d_filteredData_Re[i] = sum_re;
		d_filteredData_Im[i] = sum_im;

		d_ABS[i] = (sum_re*sum_re + sum_im*sum_im);

		i += (blockDim.x * gridDim.x);
	}
}

__global__ void parallelSum_arbitraryLen(float *a, float* sum, const int dataLength) {
	
	int i{ threadIdx.x + blockDim.x * blockIdx.x };
	if (i >= dataLength)
		return;

	int i_thr{ threadIdx.x };
	int prevPowOf2{ 1024 };

	if ((i / 1024 == gridDim.x - 1) && (dataLength % 1024 != 0)) {
		prevPowOf2 = powf(2, floorf(log2f(dataLength % 1024)));
		if (i_thr < dataLength % 1024 - prevPowOf2)
			a[i] += a[i + prevPowOf2];
		__syncthreads();
	}

	int numActive{ prevPowOf2 / 2 };
	for (int j{}; j < log2f(prevPowOf2); j++) {
		if (i_thr  < numActive)
			a[i] += a[i + numActive];
		numActive /= 2;
		__syncthreads();
	}
	if (i_thr == 0) {
		atomicAdd(sum, a[blockDim.x*blockIdx.x]);
	}
}

__global__
void NF_LoopFilter_MF(float *d_sum, const int dataLen)
{
	int denum{ dataLen };
	d_NormalizeFactor_MF = sqrtf(d_sum[0] / denum);
	if (d_NormalizeFactor_MF == 0)
		d_NormalizeFactor_MF = 1;

	d_sum[0] = 0;
}

__global__
void Normalize_MF(float *d_data_Re, float *d_data_Im, float *d_normalize_Re, float *d_normalize_Im,
	const int dataLen)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int nOut{ dataLen };
	while (i < nOut)
	{
		d_normalize_Re[i] = d_data_Re[i] / d_NormalizeFactor_MF;
		d_normalize_Im[i] = d_data_Im[i] / d_NormalizeFactor_MF;
		i += blockDim.x * gridDim.x;
	}
}

__global__
void TimingRecovery_Cubic_P1(
	const float *data_Re, const float *data_Im,
	float *LastTrVec_Re, float *LastTrVec_Im,
	float *v_vec_Re, float *v_vec_Im,
	const int dataLength)
{
	float sampVec_Re[4], sampVec_Im[4]; //0: samp0, 1: samp1, 2: samp2, 3: samp3

	int nPrevSamples, ind;
	int i{ threadIdx.x + blockIdx.x * blockDim.x };
	while (i < dataLength)
	{
		nPrevSamples = (i > 2) ? 0 : (3 - i);
		for (int n{}; n < nPrevSamples; ++n) {
			ind = (3 - nPrevSamples) + n;
			sampVec_Re[n] = LastTrVec_Re[ind];
			sampVec_Im[n] = LastTrVec_Im[ind];
		}
		for (int n{ nPrevSamples }; n < 4; ++n) {
			ind = i - (3 - n);
			sampVec_Re[n] = data_Re[ind];
			sampVec_Im[n] = data_Im[ind];
		}

		v_vec_Re[i * 4] = sampVec_Re[1];
		v_vec_Re[i * 4 + 1] = (-1 / 3.0)*sampVec_Re[0] + (-0.5)*sampVec_Re[1] + sampVec_Re[2] + (-1 / 6.0)*sampVec_Re[3];
		v_vec_Re[i * 4 + 2] = (0.5)*sampVec_Re[0] + (-1.0)*sampVec_Re[1] + (0.5)*sampVec_Re[2];
		v_vec_Re[i * 4 + 3] = (-1 / 6.0)*sampVec_Re[0] + (0.5)*sampVec_Re[1] + (-0.5)*sampVec_Re[2] + (1 / 6.0)*sampVec_Re[3];

		v_vec_Im[i * 4] = sampVec_Im[1];
		v_vec_Im[i * 4 + 1] = (-1 / 3.0)*sampVec_Im[0] + (-0.5)*sampVec_Im[1] + sampVec_Im[2] + (-1 / 6.0)*sampVec_Im[3];
		v_vec_Im[i * 4 + 2] = (0.5)*sampVec_Im[0] + (-1.0)*sampVec_Im[1] + (0.5)*sampVec_Im[2];
		v_vec_Im[i * 4 + 3] = (-1 / 6.0)*sampVec_Im[0] + (0.5)*sampVec_Im[1] + (-0.5)*sampVec_Im[2] + (1 / 6.0)*sampVec_Im[3];

		i += (blockDim.x * gridDim.x);
	}
}

void TimingRecovery_Cubic_P2_Simple(
	const float *v_vec_Re, const float *v_vec_Im,
	float *out1_Re, float *out1_Im,
	TimingRecoveryParams& params,
	const float sps,
	const float alpha, const float beta,
	const int dataLength)
{
	int j{};
	const float sps2 = sps / 2.0;
	float jitter{};
	float a{ powf(2,-alpha) }, b{ powf(2,-beta) };

	float& mu{ params.mu_final }, &ted{ params.ted_final }, &tedAcc{ params.tedAcc_final };
	int stack_tr{ params.stack_tr_final };
	bool &flagTed{ params.flagTed_final };
	float &val1_Re{ params.val1_Re_final }, &val1_Im{ params.val1_Im_final };
	float &val2_Re{ params.val2_Re_final }, &val2_Im{ params.val2_Im_final };
	float &val0pre_Re{ params.val0pre_Re_final }, &val0pre_Im{ params.val0pre_Im_final };

	float val0_Re{}, val0_Im{};
	float val0new_Re{}, val0new_Im{};

	while (stack_tr < dataLength)
	{
		val0new_Re = v_vec_Re[stack_tr * 4] + mu*(v_vec_Re[stack_tr * 4 + 1] + mu*(v_vec_Re[stack_tr * 4 + 2] + mu*v_vec_Re[stack_tr * 4 + 3]));
		val0new_Im = v_vec_Im[stack_tr * 4] + mu*(v_vec_Im[stack_tr * 4 + 1] + mu*(v_vec_Im[stack_tr * 4 + 2] + mu*v_vec_Im[stack_tr * 4 + 3]));

		val0_Re = val0new_Re;
		val0_Im = val0new_Im;

		if (flagTed)
		{
			ted = val1_Re*(val2_Re - val0_Re) + val1_Im*(val2_Im - val0_Im);
			flagTed = false;
			out1_Re[j] = val0_Re;
			out1_Im[j] = val0_Im;
			j++;
		}
		else
		{
			ted = 0;
			flagTed = true;
		}

		jitter = a*ted + b*tedAcc;
		tedAcc += ted;
		mu += (sps2 + jitter);
		val2_Im = val1_Im;
		val2_Re = val1_Re;
		val1_Im = val0_Im;
		val1_Re = val0_Re;
		stack_tr = stack_tr + floor(mu);
		mu = mu - floor(mu);
	}

	params.stack_tr_final = stack_tr - dataLength;
	params.tr_output_len = j;
}

void PLL_Simple(
	float *inOut_re,
	float *inOut_im,
	PllParams &params,
	const int PowerQPSK,
	const float alpha, const float beta,
	const bool isMSK, const int dataLength)
{
	// ----- Local Value -------- //
	float Costas_Eq;
	float sign_samp_Re;
	float sign_samp_Im;
	float jitter;
	float sampn_Re = 1;
	float sampn_Im = 0;
	float samp_estim_Re;
	float samp_estim_Im;
	float samp_Re;
	float samp_Im;
	float sampn_Re_tmp;
	int i{ 0 };
	float ped;
	//
	bool isbpsk{ false };
	// ----------- Set Param -------- //
	const float a{ powf(2,-alpha) };
	const float b{ powf(2,-beta) };
	int npow{ PowerQPSK };
	// ------ Last Block Parameters Loading --------- //
	float &phi_estim{ params.phi_estim_final };
	float &ped_acc{ params.ped_acc_final };

	if (npow == 0)
	{
		isbpsk = true;
		npow = 1;
	}

	while (i < dataLength)
	{
		sampn_Re = 1;
		sampn_Im = 0;
		samp_Re = (inOut_re[i] * cosf(-2.0 * PI*phi_estim)) - (inOut_im[i] * sinf(-2.0 * PI*phi_estim));
		samp_Im = (inOut_re[i] * sinf(-2.0 * PI*phi_estim)) + (inOut_im[i] * cosf(-2.0 * PI*phi_estim));

		for (int o = 0; o < npow; o++)
		{
			sampn_Re_tmp = sampn_Re*samp_Re - sampn_Im*samp_Im;
			sampn_Im = sampn_Im*samp_Re + sampn_Re*samp_Im;
			sampn_Re = sampn_Re_tmp;
		}

		sign_samp_Re = (((float)(sampn_Re > 0) - 0.5) * 2);
		sign_samp_Im = (((float)(sampn_Im > 0) - 0.5) * 2 * (-1));

		if (isbpsk)
			sign_samp_Im = 0;

		Costas_Eq = ((sampn_Re*sign_samp_Im) + (sampn_Im*sign_samp_Re)) / sqrtf(2);

		ped = Costas_Eq / npow;
		jitter = a*ped + b*ped_acc;
		ped_acc += ped;
		phi_estim += jitter;
		if (phi_estim > 1)
			phi_estim -= 1;
		else if (phi_estim < 0)
			phi_estim += 1;

		samp_estim_Re = samp_Re*cosf(-jitter*(PI * 2)) - samp_Im*sinf(-jitter*(PI * 2));
		samp_estim_Im = samp_Re*sinf(-jitter*(PI * 2)) + samp_Im*cosf(-jitter*(PI * 2));

		inOut_re[i] = samp_estim_Re;
		//if (isMSK)
		//	PLL_Out_Im[i] = 0;
		//else
		//	PLL_Out_Im[i] = samp_estim_Im;
		inOut_im[i] = (!isMSK) * samp_estim_Im;

		i++;
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
	
	cudaMemcpyToSymbol(c_matchFilter_Coef, sdr_params.matchedFilter_coeffs.data(), sdr_params.matchedFilter_len * sizeof(float));
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
	resampler_calc_step_host(n_threads_resample, n_blocks_resample, sig_params.sps, sdr_params.sps_final);
	int i{};
	while ((i + 1) * sdr_params.sizeOfFrame <= num_elements) {

		std::cout << "------------------ frame " << i + 1 << " -----------------\n";

		fileReader.readBinData(sdr_params.input, sdr_params.sizeOfFrame);
		separateIQ_CPU(sdr_params.input_re, sdr_params.input_im, sdr_params.input, sdr_params.sizeOfFrame);
		cudaMemcpyAsync(sdr_params.d_ddcIn_re, sdr_params.input_re, sizeof(float) * sdr_params.sizeOfFrame, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(sdr_params.d_ddcIn_im, sdr_params.input_im, sizeof(float) * sdr_params.sizeOfFrame, cudaMemcpyHostToDevice);

		gpuErrchk();

		Baseband << < 12, 1024 >> >(sdr_params.d_ddcIn_re,
			sdr_params.d_ddcIn_im,
			sdr_params.d_ddcOut_re + sdr_params.filterbb_len - 1,
			sdr_params.d_ddcOut_im + sdr_params.filterbb_len - 1,
			sdr_params.sizeOfFrame,
			sig_params.frequency);

		SetFreq_init << <1, 1 >> >(sdr_params.sizeOfFrame, sig_params.frequency);

		gpuErrchk();

		//BasebandFilter << <12, 1024 >> >(sdr_params.d_ddcOut_re, sdr_params.d_ddcOut_im,
		//	sdr_params.d_bbFilterOut_re + 4, sdr_params.d_bbFilterOut_im + 4,
		//	sdr_params.filterbb_len, sdr_params.sizeOfFrame);

		//cudaMemcpyAsync(sdr_params.d_ddcOut_re, sdr_params.d_ddcOut_re + sdr_params.sizeOfFrame,
		//	(sdr_params.filterbb_len - 1) * sizeof(float), cudaMemcpyDeviceToDevice);

		//cudaMemcpyAsync(sdr_params.d_ddcOut_im, sdr_params.d_ddcOut_im + sdr_params.sizeOfFrame,
		//	(sdr_params.filterbb_len - 1) * sizeof(float), cudaMemcpyDeviceToDevice);
		applyFreqDomainFilter(sdr_params.d_bbFilterOut_re + 4, sdr_params.d_bbFilterOut_im + 4,
			sdr_params.d_ddcOut_re, sdr_params.d_ddcOut_im,
			sdr_params.upsampleAndFreqFilterParams);

		gpuErrchk();
		resampler << < n_blocks_resample, n_threads_resample >> >
			(sdr_params.d_resampleOut_re + sdr_params.matchedFilter_len - 1,
				sdr_params.d_resampleOut_im + sdr_params.matchedFilter_len - 1,
				sdr_params.d_outLen_resample,
				sdr_params.d_bbFilterOut_re, sdr_params.d_bbFilterOut_im,
				sig_params.sps, sdr_params.sps_final, sdr_params.sizeOfFrame);
		cudaMemcpy(&outLen_resample, sdr_params.d_outLen_resample, sizeof(int), cudaMemcpyDeviceToHost);

		gpuErrchk();

		MatchFilter << <12, 1024 >> > (sdr_params.d_resampleOut_re, sdr_params.d_resampleOut_im,
			sdr_params.d_matchedFilterOut_re, sdr_params.d_matchedFilterOut_im,
			sdr_params.d_abs2_for_sum, sdr_params.matchedFilter_len, outLen_resample);

		cudaMemcpyAsync(sdr_params.d_resampleOut_re, sdr_params.d_resampleOut_re + outLen_resample,
			(sdr_params.matchedFilter_len - 1) * sizeof(float), cudaMemcpyDeviceToDevice);

		cudaMemcpyAsync(sdr_params.d_resampleOut_im, sdr_params.d_resampleOut_im + outLen_resample,
			(sdr_params.matchedFilter_len - 1) * sizeof(float), cudaMemcpyDeviceToDevice);

		gpuErrchk();

		parallelSum_arbitraryLen << <outLen_resample / 1024 + 1, 1024 >> >(
			sdr_params.d_abs2_for_sum, sdr_params.d_sum, outLen_resample);

		NF_LoopFilter_MF << <1, 1 >> >(sdr_params.d_sum, outLen_resample);

		Normalize_MF << <12, 1024 >> >(
			sdr_params.d_matchedFilterOut_re, sdr_params.d_matchedFilterOut_im,
			sdr_params.d_normalizerOut_re, sdr_params.d_normalizerOut_im, outLen_resample);

		gpuErrchk();

		TimingRecovery_Cubic_P1 << <24, 512 >> > (
			sdr_params.d_normalizerOut_re, sdr_params.d_normalizerOut_im,
			sdr_params.d_LastTrVec_re, sdr_params.d_LastTrVec_im,
			sdr_params.d_vVec_re, sdr_params.d_vVec_im,
			outLen_resample);

		cudaMemcpyAsync(sdr_params.h_vVec_re, sdr_params.d_vVec_re, 4 * sizeof(float) * outLen_resample, cudaMemcpyDeviceToHost);
		cudaMemcpy(sdr_params.h_vVec_im, sdr_params.d_vVec_im, 4 * sizeof(float) * outLen_resample, cudaMemcpyDeviceToHost);

		gpuErrchk();

		TimingRecovery_Cubic_P2_Simple(
			sdr_params.h_vVec_re, sdr_params.h_vVec_im,
			sdr_params.tr_output_1_re, sdr_params.tr_output_1_im,
			sdr_params.tr_params, sig_params.sps, sdr_params.alpha, sdr_params.beta, outLen_resample);

		std::cout << "tr_output_len: " << sdr_params.tr_params.tr_output_len << std::endl;

		cudaMemcpyAsync(sdr_params.d_LastTrVec_re, sdr_params.d_normalizerOut_re + outLen_resample - 3, 3 * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(sdr_params.d_LastTrVec_im, sdr_params.d_normalizerOut_im + outLen_resample - 3, 3 * sizeof(float), cudaMemcpyDeviceToDevice);

		gpuErrchk();

		PLL_Simple(sdr_params.tr_output_1_re, sdr_params.tr_output_1_im,
			sdr_params.pll_params, sdr_params.npow, sdr_params.alpha, sdr_params.beta,
			false, sdr_params.tr_params.tr_output_len);

		combineIQ_CPU(sdr_params.output, sdr_params.tr_output_1_re, sdr_params.tr_output_1_im, sdr_params.tr_params.tr_output_len);
		fileWriter.writeBinData(sdr_params.output, sdr_params.tr_params.tr_output_len);

		gpuErrchk();
		i++;
	}
}