#include "ProcessingCore.h"
#include "kernels.cuh"

ProcessingCore::ProcessingCore(const SignalParams& _sig_params) {
	sig_params = _sig_params;
}

void ProcessingCore::process() {
	initialize();
	SDR_Implement(sdr_params, sig_params);
	freeMemories();
}

void ProcessingCore::initialize() {
	setBBFilterCoeffs();
	setMatchedFilterCoeffs(sig_params.rollOff);
	sig_params.frequency = sig_params.central_freq / sig_params.Fs;

	sdr_params.upsample_rate = (sig_params.sps < sdr_params.sps_final) + 1;
	sig_params.sps *= sdr_params.upsample_rate;
	sdr_params.upsampleAndFreqFilterParams.upsampleIsActive = (sdr_params.upsample_rate > 1);
	initUpsampleAndFreqFilterParams(sdr_params.upsampleAndFreqFilterParams, sdr_params.filterbb_coeffs.data(),
		sdr_params.filterbb_len, sdr_params.sizeOfFrame);

	sdr_params.tr_params.reset();
	sdr_params.pll_params.reset();

	sdr_params.input = new cuComplex[sdr_params.sizeOfFrame];
	sdr_params.input_re = new float[sdr_params.sizeOfFrame];
	sdr_params.input_im = new float[sdr_params.sizeOfFrame];

	cudaMalloc(&sdr_params.d_ddcIn_re, sdr_params.sizeOfFrame * sizeof(float));
	cudaMalloc(&sdr_params.d_ddcIn_im, sdr_params.sizeOfFrame * sizeof(float));
	cudaMalloc(&sdr_params.d_ddcOut_re, (sdr_params.sizeOfFrame + sdr_params.filterbb_len - 1) * sizeof(float));
	cudaMalloc(&sdr_params.d_ddcOut_im, (sdr_params.sizeOfFrame + sdr_params.filterbb_len - 1) * sizeof(float));

	cudaMalloc(&sdr_params.d_bbFilterOut_re, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 4) * sizeof(float));
	cudaMalloc(&sdr_params.d_bbFilterOut_im, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 4) * sizeof(float));

	cudaMalloc(&sdr_params.d_outLen_resample, sizeof(int));
	cudaMalloc(&sdr_params.d_resampleOut_re, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + sdr_params.matchedFilter_len - 1 + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_resampleOut_im, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + sdr_params.matchedFilter_len - 1 + 1000) * sizeof(float));

	cudaMalloc(&sdr_params.d_matchedFilterOut_re, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_matchedFilterOut_im, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_abs2_for_sum, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_sum, sizeof(float));
	cudaMalloc(&sdr_params.d_normalizerOut_re, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_normalizerOut_im, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	
	cudaMallocHost(&sdr_params.h_vVec_re, 4 * (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float)); //pinned memory
	cudaMallocHost(&sdr_params.h_vVec_im, 4 * (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_vVec_re, 4 * (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));
	cudaMalloc(&sdr_params.d_vVec_im, 4 * (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) * sizeof(float));

	cudaMalloc(&sdr_params.d_LastTrVec_re, 3 * sizeof(float));
	cudaMalloc(&sdr_params.d_LastTrVec_im, 3 * sizeof(float));

	sdr_params.tr_output_1_re = new float[(sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) / 2];
	sdr_params.tr_output_1_im = new float[(sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) / 2];
	sdr_params.output = new cuComplex[(sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) / 2];

	gpuErrchk();
}

void ProcessingCore::freeMemories() {
	resetUpsampleAndFreqFilterParams(sdr_params.upsampleAndFreqFilterParams);
}

void ProcessingCore::setBBFilterCoeffs() {
	readBinData(sdr_params.filterbb_coeffs, sig_params.filterbb_fileAddress); 
	sdr_params.filterbb_len = sdr_params.filterbb_coeffs.size();
}

void ProcessingCore::setMatchedFilterCoeffs(float Rcosine_alpha)
{
	int sps{ 4 };
	const int n_isi{ 32 };
	const float Pi{ 3.1415926535897f };
	int filterLen{ sps * n_isi + 1 };
	float *n = new float[filterLen];
	int idx{};
	for (float i{ -n_isi / 2.f }; i <= n_isi / 2.f; i += 1 / static_cast<float>(sps))
		n[idx++] = i;

	sdr_params.matchedFilter_len = filterLen;
	sdr_params.matchedFilter_coeffs.resize(sdr_params.matchedFilter_len);
	for (size_t i{}; i < filterLen; i++) {
		if (n[i] == 0)
			sdr_params.matchedFilter_coeffs[i] = 1 - Rcosine_alpha + 4 * Rcosine_alpha / Pi;

		else if (fabsf(fabsf(n[i]) - 1 / 4.f / Rcosine_alpha) <= 1e-6) //correct?
			sdr_params.matchedFilter_coeffs[i] = Rcosine_alpha / sqrtf(2) * ((1 + 2 / Pi) * sinf(Pi / 4 / Rcosine_alpha) + (1 - 2 / Pi) * cosf(Pi / 4 / Rcosine_alpha));

		else
			sdr_params.matchedFilter_coeffs[i] = (4 * Rcosine_alpha / Pi) / (1 - powf(4 * Rcosine_alpha * n[i], 2)) * (cosf((1 + Rcosine_alpha)*Pi*n[i]) + sinf((1 - Rcosine_alpha)*Pi*n[i]) / (4 * Rcosine_alpha*n[i]));
	}
	delete[] n;
}
