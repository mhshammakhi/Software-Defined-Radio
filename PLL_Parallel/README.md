
# PLL_Parallel

<p align="justify"> 
This folder encompasses the CUDA implementation of two GPU-based Phased-Locked Loop (PLL) algorithms designed specifically for PSK signals. These algorithms provide efficient and parallel processing capabilities, enhancing the performance of PLL in GPU environments.
</p>

  - The Forward-Backward Algorithm
  - The Intra-Frame Parallelism Algorithm
  - The Enhanced FB-PLL Algorithm

<div align="justify"> 
 The folder includes static library files (.lib) for each algorithm, designed for Windows. These libraries, built using Visual Studio 2015, are available in both debug and release builds (x64 only). Additionally, an executable file is provided for testing purposes, which outputs symbols and processing rates. For instructions on running the executable, please refer to the README file inside the PLL_Parallel directory.

The first algorithm implements a parallelized PLL for multiple frames. To gain a comprehensive understanding of the methodology and view the results, please refer to [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4100444) article.

In the second algorithm, parallelism is employed within a single frame, allowing for parallel compensation of multiple consecutive symbols. This approach differs from the first algorithm, which applies parallelism across multiple frames.

The third algorithm achieves significantly faster processing speeds by leveraging efficient memory management and shorter frame requirements, ensuring compatibility with a wide range of GPUs while reducing overall memory usage.
</p>

## The implementation of GPU based Phased Locked Loop for PSK signals in CUDA.

To fully understand the details of this work and to see the results, please see [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4100444) article:
Shammakhi, Mohammad Hasan, et al. "Gb/Sec Frame-Based Phase Locked Loop on Gpu." Available at SSRN 4100444.

- [x] In ForwardBackward project there are:

    1. CudaLib of PLL
    2. ExecuteFiles for testing
    3. MATLAB Code to understand the theory structure. Also, you can validate the algorithm by MATLAB code.
    
## The implementation of GPU based Enhanced FB-PLL in CUDA.

To fully understand the details of this work and to see the results, You can see **"Adaptive Forward-Backward Phased Locked Loop."** paper

- [x] In Enhanced-FB directory there are:

    1. CudaLib of PLL
    2. MATLAB Code to understand the theory structure. Also, you can validate the algorithm by the MATLAB code.
    
    
