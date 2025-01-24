This project is created by : Mohammad Hasan Shammakhi

# The implementation of GPU based Phased Locked Loop for PSK signals in CUDA.

To fully understand the details of this work and to see the results, You can see **"Adaptive Forward-Backward Phased Locked Loop."** paper

- [x] In this project there are:

    1. CudaLib of PLL
    2. ExecuteFiles for testing
    3. matlabCode to understand the theory structure. Also, you can validate the algorithm by matlab code.
    
    
## CudaLib of PLL

This is the generated lib for using **Enhanced FB-PLL** in your system. 

## ExecuteFiles for testing

You can test the forward-backward PLL by running the exe file `.\PLL.exe --signal-file ".\\pll_input.bin" --modulation "QPSK" --frame-length 80000 --frame-count 20`
    - you can set input file address by setting `--signal-file` 
    - arg `--modulation` sets the modulation type.
    - each frame length is set by `--frame-length`
    - number of parallel frame can set by `--frame-count`
all of these parameters defenition can be found in **"Adaptive Forward-Backward Phased Locked Loop."**.    

## Matlab Code

In this folder there are four **m file** `createInput.m`, `main.m`, `PLL_V3_Enhanced_FB.m` and `PLL_V3_FB`.
By using `createInput.m` you can create signal with the desired value of frequency and phase offset
In `main.m` we compensate frequency and phase offset by using  `PLL_V3_Enhanced_FB.m`
and `PLL_V3_EFB.m` is the implemention of Enhanced FB PLL

## Open source CUDA code

The CUDA source file will upload after a financial support or contract.
