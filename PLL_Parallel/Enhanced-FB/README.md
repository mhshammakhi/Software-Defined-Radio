This project is created by : Mohammad Hasan Shammakhi

# The implementation of GPU based Enhanced FB-PLL in CUDA.

To fully understand the details of this work and to see the results, You can see **"Adaptive Forward-Backward Phased Locked Loop."** paper

- [x] In this project there are:

    1. CudaLib of PLL
    2. MATLAB code to understand the theory structure. Also, you can validate the algorithm by MATLAB code.
    
    
## CudaLib of PLL

This is the generated lib for using **Enhanced FB-PLL** in your system. 

## MATLAB Code

In this folder there are four **m file** `createInput.m`, `main.m`, `PLL_V3_Enhanced_FB.m` and `PLL_V3_FB`.
By using `createInput.m` you can create signal with the desired value of frequency and phase offset
In `main.m` we compensate frequency and phase offset by using  `PLL_V3_Enhanced_FB.m`
and `PLL_V3_EFB.m` is the implemention of Enhanced FB PLL

## Open source CUDA code

The CUDA source file will upload after a financial support or contract.
