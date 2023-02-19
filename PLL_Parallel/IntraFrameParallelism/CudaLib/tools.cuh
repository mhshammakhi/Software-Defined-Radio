#ifndef TOOLS_H
#define TOOLS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<cmath>

#define PI 3.14159265359f

#define Max_PLL_nPar 32

struct TR_PLL_Input {
	bool flagTed_final{ true };
	float mu_final{}, ted_final{}, tedAcc_final{};
	float mu_frac_final{}, mu_pre_final{};
	float LastTrVec_Re[3]{}, LastTrVec_Im[3]{};
	float val1_Re_final{}, val1_Im_final{}, val2_Re_final{}, val2_Im_final{};
	int stack_tr_final{};
	int numberOfTr{};
	int d_nValid_vec{};
	float phi_estim_final{}, ped_acc_final{};
	float phi_estim_final_vec[Max_PLL_nPar]{};
	float val0pre_Re_final{}, val0pre_Im_final{};
	float val_Re_final{}, val_Im_final{};
	int iter{};

	//======= OQPSK Specific======//
	float val_final_final_Re{}, val_final_final_Im{};
	float PPA_final{};
	//phi_estim_vec is kinda duplicate but whatever
	float pll_out_pow_Re_final{}, pll_out_pow_Im_final{};
	//===========common==========//
	int PLL_nPar, TR_nPar, nPar_OQPSK;
};

//====================== PLL kernels ======================//	
__global__
void PLL_Parallel_Latest(
	const float *d_data_Re,
	const float *d_data_Im,
	float *d_PLL_Out_Re,
	float *d_PLL_Out_Im,
	const int PowerQPSK,
	const float a, const float b,
	const int n_not_processed,
	const bool isMSK, TR_PLL_Input *pllParams);

#endif // ! TOOLS_H