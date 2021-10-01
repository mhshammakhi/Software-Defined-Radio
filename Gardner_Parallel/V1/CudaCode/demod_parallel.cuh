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
	float val1_Re_final{}, val1_Im_final{}, val2_Re_final{}, val2_Im_final{};
	int stack_tr_final{};
	int numberOfTr{};
	int d_nValid_vec{};
	bool flag_1{true};
	float phi_estim_final{}, ped_acc_final{};
};

__global__
void PLL_Parallel(
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
void TimingRecovery_Par(
        const float *, const float *,
        float *, float *,
        float *, float *,
        const float,
        const float ,const float,
        const int,
        const int,
		TR_PLL_Input*);

#endif // ! PLL_AND_TR_H