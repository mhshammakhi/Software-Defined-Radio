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
	sig_params.frequency = sig_params.central_freq / sig_params.Fs;

	sdr_params.upsample_rate = (sig_params.sps < sdr_params.sps_final) + 1;
	sig_params.sps *= sdr_params.upsample_rate;
	sdr_params.upsampleAndFreqFilterParams.upsampleIsActive = (sdr_params.upsample_rate > 1);
	initUpsampleAndFreqFilterParams(sdr_params.upsampleAndFreqFilterParams, sdr_params.filterbb_coeffs.data(),
		sdr_params.filterbb_len, sdr_params.sizeOfFrame);


	sdr_params.input = new cuComplex[sdr_params.sizeOfFrame];
	sdr_params.input_re = new float[sdr_params.sizeOfFrame];
	sdr_params.input_im = new float[sdr_params.sizeOfFrame];

	cudaMalloc(&sdr_params.d_ddcIn_re, sdr_params.sizeOfFrame * sizeof(float));
	cudaMalloc(&sdr_params.d_ddcIn_im, sdr_params.sizeOfFrame * sizeof(float));
	cudaMalloc(&sdr_params.d_ddcOut_re, (sdr_params.sizeOfFrame + sdr_params.filterbb_len - 1) * sizeof(float));
	cudaMalloc(&sdr_params.d_ddcOut_im, (sdr_params.sizeOfFrame + sdr_params.filterbb_len - 1) * sizeof(float));
	std::cout << (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 4) << std::endl;
	cudaMalloc(&sdr_params.d_bbFilterOut_re, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 4) * sizeof(float));
	cudaMalloc(&sdr_params.d_bbFilterOut_im, (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 4) * sizeof(float));
	sdr_params.h_out_re = new float[sdr_params.sizeOfFrame];
	sdr_params.h_out_im = new float[sdr_params.sizeOfFrame];
	std::cout << (sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000) << std::endl;
	sdr_params.output = new cuComplex[(sdr_params.upsample_rate * sdr_params.sizeOfFrame + 1000)];

	gpuErrchk();
}

void ProcessingCore::freeMemories() {
	resetUpsampleAndFreqFilterParams(sdr_params.upsampleAndFreqFilterParams);
}

void ProcessingCore::setBBFilterCoeffs() {
	readBinData(sdr_params.filterbb_coeffs, sig_params.filterbb_fileAddress); 
	sdr_params.filterbb_len = sdr_params.filterbb_coeffs.size();
}


