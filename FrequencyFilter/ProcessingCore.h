#pragma once

#include"definition.h"
#include"utils.h"
#include "freq_filter.cuh"

class ProcessingCore
{
public:
	ProcessingCore(const SignalParams& _sig_params);

	void process();

private:
	SignalParams sig_params;
	SDR_Params sdr_params;

	void initialize();
	void freeMemories();
	void setBBFilterCoeffs();
};

