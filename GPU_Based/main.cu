
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include"ProcessingCore.h"
#include"utils.h"
#include"definition.h"
#include<tuple>
#include <tuple>
#include <iostream>
#include <string>
#include <stdexcept>



std::tuple<std::string, std::string, std::string> parseCommandLineArguments(int argc, char** argv);
int main(int argc, char** argv)
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
	std::tuple<std::string, std::string, std::string> a;
	a = parseCommandLineArguments(argc, argv);
	sig_params.inputFileAddress = std::get<0>(a);
	sig_params.outputFileAddress = std::get<1>(a);
	sig_params.filterbb_fileAddress = std::get<2>(a);

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

std::tuple<std::string, std::string, std::string> parseCommandLineArguments(int argc, char** argv)
{

	std::string inputFileAddress = "./InputFiles/signal.bin";
	std::string outputFileAddress = "SDR_Output.bin";
	std::string filterbb_fileAddress = "./InputFiles/bbFilterCoeffs.bin";




	for (int i = 1; i < argc; i++) {
		if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
			char arg = argv[i][1];
			unsigned int* toSet = 0;
			switch (arg) {
			case 'i':
				inputFileAddress = argv[i] + 3;
				break;
			case 'f':
				filterbb_fileAddress = argv[i] + 3;
				break;
			case 'o':
				outputFileAddress = argv[i] + 3;
				break;
			}
		}
	}
	return{ inputFileAddress, outputFileAddress, filterbb_fileAddress };
}