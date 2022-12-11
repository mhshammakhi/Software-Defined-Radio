#pragma once

#include<string>
#include<vector>
#include<cuComplex.h>
#include<cufft.h>

struct FreqFilterParams {
	int n_fft;
	int stateLen, blockLen, n_blocks;
	int inputLen;
	bool has_extra_block;

	cuComplex *d_data, *d_data_expanded;
	cuComplex *d_middle;
	cuComplex *d_out, *d_out_expanded;
	float *d_filterCoeffs;
	cuComplex *d_filterCoeffsFft;
	cuComplex *d_filterCoeffH_SymbolAddress;

	cufftHandle fftPlanForCoeffs;
	cufftHandle fftPlanForInput, ifftPlanForInput;
};

struct UpsampleAndFreqFilterParams {
	FreqFilterParams freqFilterParams;

	cuComplex *d_out, *d_out_expanded;
	cuComplex *d_filterCoeffsFft;
	cuComplex *d_filterCoeffH_SymbolAddress;

	bool upsampleIsActive;
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
	const float sps_final{ 4 };
	const int sizeOfFrame{ 2 * 1024 * 1024 };
	float npow{ 1 };
	int upsample_rate;

	UpsampleAndFreqFilterParams upsampleAndFreqFilterParams;

	std::vector<float> filterbb_coeffs, matchedFilter_coeffs;
	int filterbb_len, matchedFilter_len;

	cuComplex *input;
	float *input_re, *input_im;
	float *d_ddcIn_re, *d_ddcIn_im;
	float *d_ddcOut_re, *d_ddcOut_im;
	float *d_bbFilterOut_re, *d_bbFilterOut_im;
	float * h_out_re, * h_out_im;
	cuComplex *output;
};
