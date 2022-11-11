
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include"ProcessingCore.h"
#include"utils.h"
#include"definition.h"


int main()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	//------------------- Set Input Params ------------------
	SignalParams sig_params;
	sig_params.inputFileAddress = "./signal_Fc1_SPS4_ModuQPSK_SNR15_float.bin";
	sig_params.outputFileAddress = "SDR_Output.bin";
	sig_params.filterbb_fileAddress = "./InputFiles/bbFilterCoeffs.bin";

	sig_params.sps = 4;
	sig_params.Fs = sig_params.sps;
	sig_params.Rs = 1;
	sig_params.BW = 1.3;
	sig_params.central_freq = 0.4;
	sig_params.rollOff = 0.25;

	ProcessingCore ProcessingCore_Obj{ sig_params };
	ProcessingCore_Obj.process();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	std::cout << "Finished!\n";
    return 0;
}
