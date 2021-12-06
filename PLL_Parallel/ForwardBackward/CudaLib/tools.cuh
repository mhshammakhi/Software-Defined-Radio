#ifndef PLL_AND_TR_H
#define PLL_AND_TR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<cmath>

#define pi 3.14159265359f

#define PLL_nPar 32
#define TR_nPar 32

struct TR_PLL_Input {
	bool flagTed_final{ true };
	float mu_final{}, muAcc_final{}, ted_final{}, tedAcc_final{};
	float mu_frac_final{}, mu_pre_final{};
	float LastTr_Re{}, LastTr_Im{};
	float LastTr1_Re{}, LastTr1_Im{}, LastTr2_Re{}, LastTr2_Im{};
	float val1_Re_final{}, val1_Im_final{}, val2_Re_final{}, val2_Im_final{};
	int stack_tr_final{};
	int numberOfTr{};
	int d_nValid_vec{};
	float phi_estim_final{}, ped_acc_final{};
	float phi_estim_final_vec[PLL_nPar]{};
	float val0pre_Re_final{}, val0pre_Im_final{};
	float val_Re_final{}, val_Im_final{};
	int iter{};
};

//====================== PLL kernels ======================//	
__global__
void PLL_Parallel_Latest(
	float *,
	float *,
	float *,
	float *,
	const int,
	const float, const float,
	const int,
	const int,
	TR_PLL_Input*);


__global__
void PLL_Simple(
	float *,
	float *,
	float *,
	float *,
	const int,
	const float, const float,
	const int,
	const int,
	TR_PLL_Input*);

//====================== TimingRecovery kernels =====================//
__global__
void TimingRecovery_Cubic_P1(
	const float *, const float *,
	float *, float *,
	const int, TR_PLL_Input*);

__global__
void TimingRecovery_Cubic_P2_Simple(
	const float *, const float *,
	const float *, const float *,
	float *, float *,
	float *, float *,
	const float ,
	const float, const float,
	const int, const bool,
	const int, TR_PLL_Input*);

__global__
void TimingRecovery_Cubic_P2_Par(const float *, const float *,
	const float *, const float *,
	float *, float *,
	float *, float *,
	const float,
	const float, const float,
	const int,
	const int, TR_PLL_Input*);

//================================ Wrappers ================================//
void TimingRecovery(const float,
	const float *, const float *,
	float *, float *,
	float *, float *,
	float *, float *,
	const float,
	const float, const float,
	const int,
	const int, const bool,
	TR_PLL_Input*);

void PLL(const float,
	float *,
	float *,
	float *,
	float *,
	const int,
	const float, const float,
	const int,
	const int,
	TR_PLL_Input*);


#endif // ! PLL_AND_TR_H