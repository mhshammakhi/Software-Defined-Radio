This project is created by : Mohammad Hasan Shammakhi

# The implementation of GPU based Phased Locked Loop for PSK signals in CUDA.

<div align="justify">
The information required for the PLL block to perform the algorithm are the signal file and the modulation type. The input signal to the PLL kernel is a binary (.bin) file containing complex float symbols with real and imaginary parts saved successively in the file as: x_re[0], x_im[0], x_re[1], x_im[1], etc. The modulation type can be one of the following:
</div>

- BPSK
- QPSK
- 8PSK

Even though this kernel was mainly aimed at PSK family signals, it works with other modulation families such as 16QAM, 64QAM, 16APSK, and 32APSK.

In this project there are the following folders:

    1. CudaLib containing .lib files
    2. ExecuteFiles containing the .exe file
    
    
## CudaLib of PLL

In this folder, the static library files for windows (.lib) are made available.

## ExecuteFiles for testing

<p align="justify">
You can test the intra-frame parallel PLL by running the RunMe.Bat file in the folder. The inputs that can be given to the .exe file through the command line are not stable yet. The program does not support argument naming for now. You will have to follow a strict order for giving the inputs to the command line. As the RunMe.Bat file suggests, the first argument is the input file address and the second is the modulation type.
</p>

## Open source CUDA code

The CUDA source file will upload after a financial support or contract.
