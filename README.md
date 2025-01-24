# Software Defined Radio (SDR)
Created by Mohammad Hasan Shammakhi

This project contains:
1. [x] CPU-Based SDR
2. [x] GPU-Based SDR
3. [ ] FPGA-Based SDR

The project consists of multiple directories, each serving a distinct purpose. Here's a concise overview of the main directories in this repository.

## FrequencyFilter

<!--- <p style="text-align: justify;"> -->
<p align="justify"> 

In this directory, you'll find the source code for a frequency-domain filter. Applying the filter on a signal becomes highly efficient in the frequency domain, especially as the input signal length increases. Additionally, our implementation includes an optional upsampling by a factor of two, which was required for our SDR system with a minimum SPS of 4. The function utilizes the cuFFT library provided by CUDA. 
</p>

## Gardner_Parallel

<p align="justify"> 
In this folder, you'll find two versions of timing recovery based on the Gardner algorithm.
</p>

  - The method which is explain in [**GPU-Based Parallel Algorithm for Wideband Signal Timing Recovery**](https://www.turcomat.org/index.php/turkbilmat/article/view/12005 "GPU-Based Parallel Algorithm for Wideband Signal Timing Recovery")
  - The forward-backward method which will be published soon in an article.

## OtherBlocks

<p align="justify"> 
This directory contains commonly used blocks in SDR systems, excluding the demodulation section. Currently, it includes a digital downconverter (DDC) referred to as *Baseband* in the source code, and a time-domain filter. While the frequency domain version is available, the time domain version is simpler to implement and offers satisfactory performance in many applications.
</p>

## PLL_Parallel

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

## Signal

In this folder, we provide a selection of signals that can be used to test and evaluate the performance of the algorithms. These signals serve as representative examples to assess the effectiveness and accuracy of the implemented algorithms. By using these test signals, users can verify the functionality and suitability of the algorithms for their specific requirements.

**For more details or inquiries, feel free to reach out to me at <ins>mh.shammakhi@gmail.com</ins>. I am available to provide further information or answer any questions you may have.**
