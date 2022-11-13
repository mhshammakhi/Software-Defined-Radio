This project is created by : Mohammad Hasan Shammakhi

# The implementation of GPU based Phased Locked Loop for PSK signals in CUDA.

To fully understand the details of this work and to see the results, please see the [this][https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4100444] article:

Shammakhi, Mohammad Hasan, et al. "Gb/Sec Frame-Based Phase Locked Loop on Gpu." Available at SSRN 4100444.

- [x] In this project there are:

    1. CudaLib of PLL
    2. ExecuteFiles for testing
    3. matlabCode to understand the theory structure. Also, you can validate the algorithm by matlab code.
    
    
## CudaLib of PLL

This is the generated lib for using **forward backward PLL** in your system. 

## ExecuteFiles for testing

You can test the forward-backward PLL by running the exe file `.\PLL.exe --signal-file ".\\pll_input.bin" --modulation "QPSK" --frame-length 80000 --frame-count 20`
    - you can set input file address by setting `--signal-file` 
    - arg `--modulation` sets the modulation type.
    - each frame length is set by `--frame-length`
    - number of parallel frame can set by `--frame-count`
all of these parameters defenition can be found in **"Gb/Sec Frame-Based Phase Locked Loop on Gpu."**.    

    

