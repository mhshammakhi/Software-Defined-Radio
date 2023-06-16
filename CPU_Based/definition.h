#pragma once

#include <string>
#include <vector>

struct FreqFilterParams {
	int n_fft;
	int stateLen, blockLen, n_blocks;
	int inputLen;
	bool has_extra_block;

	// fftw_complex *d_data, *d_data_expanded;
	// fftw_complex *d_middle;
	// fftw_complex *d_out, *d_out_expanded;
	float *d_filterCoeffs;
	// fftw_complex *d_filterCoeffsFft;
	// fftw_complex *d_filterCoeffH_SymbolAddress;

	// cufftHandle fftPlanForCoeffs;
	// cufftHandle fftPlanForInput, ifftPlanForInput;
};

struct UpsampleAndFreqFilterParams {
	FreqFilterParams freqFilterParams;

	// fftw_complex *d_out, *d_out_expanded;
	// fftw_complex *d_filterCoeffsFft;
	// fftw_complex *d_filterCoeffH_SymbolAddress;

	bool upsampleIsActive;
};

struct TimingRecoveryParams {
	float mu_final, ted_final, tedAcc_final;
	int stack_tr_final;
	bool flagTed_final;
	float val1_Re_final, val1_Im_final;
	float val2_Re_final, val2_Im_final;
	float val0pre_Re_final, val0pre_Im_final;
	int tr_output_len;

	TimingRecoveryParams() {
		reset();
	}
	void reset() {
		mu_final = ted_final = tedAcc_final = 0;
		stack_tr_final = 0;
		flagTed_final = true;
		val1_Re_final = val1_Im_final = 0;
		val2_Re_final = val2_Im_final = 0;
		val0pre_Re_final = val0pre_Im_final = 0;
		tr_output_len = 0;
	}
};

struct PllParams {
	float phi_estim_final, ped_acc_final;

	PllParams() {
		reset();
	}
	void reset() {
		phi_estim_final = ped_acc_final = 0;
	}
};

struct SignalParams {
	std::string filterbb_fileAddress;
	std::string inputFileAddress;
	std::string outputFileAddress;
	float sps, Fs, Rs;
	float BW;
	float central_freq, frequency;
	float rollOff;
};

struct SDR_Params {
	const float alpha{ 10 }, beta{ 20 };
	const float sps_final{ 4 };
	const int sizeOfFrame{ 2 * 1024 * 1024 };
	float npow{ 1 };
	int upsample_rate;

	UpsampleAndFreqFilterParams upsampleAndFreqFilterParams;

	std::vector<float> filterbb_coeffs, matchedFilter_coeffs;
	int filterbb_len, matchedFilter_len;
	TimingRecoveryParams tr_params;
	PllParams pll_params;

	// fftw_complex *input;
	float *input_re, *input_im;
	float *d_ddcIn_re, *d_ddcIn_im;
	float *d_ddcOut_re, *d_ddcOut_im;
	float *d_bbFilterOut_re, *d_bbFilterOut_im;
	float *d_resampleOut_re, *d_resampleOut_im;
	int *d_outLen_resample;

	float *d_matchedFilterOut_re, *d_matchedFilterOut_im;
	float *d_abs2_for_sum;
	float *d_sum;
	float *d_normalizerOut_re, *d_normalizerOut_im;

	float *h_vVec_re, *h_vVec_im;
	float *d_vVec_re, *d_vVec_im;
	float *d_LastTrVec_re, *d_LastTrVec_im;
	float *tr_output_1_re, *tr_output_1_im;
	// fftw_complex *output;
};
