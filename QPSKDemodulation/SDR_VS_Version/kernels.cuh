#pragma once

#include<cuComplex.h>
#include "definition.h"
#include "utils.h"
#include "freq_filter.cuh"

void separateIQ_CPU(float *data_re, float *data_im, const cuComplex *data_comp, const int& dataLen);
void combineIQ_CPU(cuComplex *data_comp, const float *data_re, const float *data_im, const int& dataLen);
void SDR_Implement(SDR_Params& sdr_params, const SignalParams& sig_params);