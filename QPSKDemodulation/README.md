This project is created by : Mohammad Hasan Shammakhi

# The implementation of GPU based demodulation system for QPSK signals in CUDA.



In this project the qpsk demodulation system is implemented. This project can be used for other Modulation in PSK family by little changes.

A short signal is uploded in this folder but in order to have better performance you should generate a long signal to validate the system.

To fully understand the details of this work and to see the results, please watch the "Capstone Presentaiton" video.

In this project there are DDC block, Filter BaseBand (Frequency domain filter based on cufft), resampler, Matched filter, Power normalizer, Timing recovery and PLL.
# System requirements:
the source files are compatible with all visual studio versions but if you want to use my .sln file
1. Install Visual Studio 2015 update 3  (select c++ option)
2. Install CUDA 9.0

enjoy the project by using .sln file.

# Source Codes:

The source codes are available in https://github.com/mhshammakhi/SDR_GPU/tree/main/QPSKDemodulation/SDR_VS_Version

Table, like this one :

File Name  | Description
------------- | -------------
Input Files Folder  | Contains filter coefficient binary file
ProcessingCore.cpp .h  |  Initialization of gpu arrays and blocks and controling the system parameters
definition.h  |  It contains the definition of all structures which we need in processing core and kernels
freq_filter.cu .cuh|  base band filter kernels (in general all kernels which is designed for frequency based filtering by overlap add method)
kernels.cu .cuh | DDC, filter, resampler, power normalizer, matched filter and all other kernels we have as the processing blocks
main.m | For defining controling and caling the main parameters and functions of system.
utils.h | some useful functions for reading writing and other general purposes

```
