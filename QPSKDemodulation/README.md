This project is created by : Mohammad Hasan Shammakhi

# The implementation of GPU based demodulation system for QPSK signals in CUDA.



In this project the qpsk demodulation system is implemented. This project can be used for other Modulation in PSK family by little changes.

To fully understand the details of this work and to see the results, please watch the "Capstone Presentaiton" video.

- [x] In this project there are DDC block, Filter BaseBand (Frequency domain filter based on cufft), resampler, Matched filter, Power normalizer, Timing recovery and PLL.
# System requirements:
the source files are compatible with all visual studio versions but if you want to use my .sln file
1. Install Visual Studio 2015 update 3  (select c++ option)
2. Install CUDA 9.0

enjoy the project by using .sln file.

Also, the project is tested by visual studio 2019 and CUDA 10.1.

# Source Codes:

The source codes are available in [here](https://github.com/mhshammakhi/SDR_GPU/tree/main/QPSKDemodulation/SDR_VS_Version "Source Codes")

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



# Other Files:

In "Capstone Presentation.mp4" I present more details about processing blocks and code details.
The presentation file can be found in "Capstone Presentation.pdf".
"signal.bin" is a short signal which is uploded in this folder but in order to have better performance you should generate a long signal to validate the system.

# Proof of execution artifacts:

If you set the correct address for input signal and if it has at least 2M samples the code will run and you can see the number of symbol in each block
If you have a shor signal with short length with less than 2 Mega samples you can change the frame_size to run the project.

You can see one example [here](https://github.com/mhshammakhi/SDR_GPU/tree/main/QPSKDemodulation/output/Capture.PNG "Run code Example"):

![Alt text](https://github.com/mhshammakhi/SDR_GPU/tree/main/QPSKDemodulation/output/Capture.PNG "Run code Example")

![Alt text](https://github.com/mhshammakhi/SDR_GPU/tree/main/QPSKDemodulation/output/RunningCode.gif)

# Refrences:

1. E. Grayver, Implementing software defined radio, Springer Science & Business Media, 2012.
2. M. H. Shammakhi, P. H. Faraji, M. M. Bidhandi, M. Hosseinzadeh, Gpu-based parallel algorithm for wideband signal timing recovery, Turkish Journal of Computer and Mathematics Education (TURCOMAT) 13 (1) (2022) 190–197.
3. C. Nvidia, Compute unified device architecture programming 505 guide.
4. J. Cheng, M. Grossman, T. McKercher, Professional CUDA c programming, John Wiley & Sons, 2014.
