# SDR_GPU
This project is created by: Mohammad Hasan Shammakhi

The project containts several directories, each serving a different purpose. This is a brief overview of the main directories in this repository.

## FrequencyFilter

<p style="text-align: justify;">
In this directory, the source code of a frequency-domain filter is presented. As the length of the input signal increases, applying a filter on the signal will become much more efficient in the frequency domain compared to the time domain. Furthermore, our implementation can also apply an upsampling of factor two if necessary (as part of our SDR system, we required an SPS of at least 4, hence the system applies a upsampling for SPS values less than 4). The proposed function uses the cuFFT library by CUDA. 
</p>

## Gardner_Parallel

<p style="text-align: justify;">
In this folder we have provided two versions of timing recovery based on the Gardner algorithm:
</p>

  - The method which is explain in [**GPU-Based Parallel Algorithm for Wideband Signal Timing Recovery**](https://www.turcomat.org/index.php/turkbilmat/article/view/12005 "GPU-Based Parallel Algorithm for Wideband Signal Timing Recovery")
  - The forward-backward method which will be published soon in an article.

## OtherBlocks

<p style="text-align: justify;">
This directory is intended for blocks that are commonly used in SDR systems other than the demodulation section. So far we have provided a digital downconverter (DDC), which we call *Baseband* in the source code, and a time-domain filter. Even though we have already put the frequency domain version, the time domain version is much easier to implement and has acceptable performance in many applications.
</p>

## PLL_Parallel

<p style="text-align: justify;">
This folder contains the implementation of two GPU-based Phased-Locked Loop algorithms for PSK signals in CUDA:
</p>

  - The Forward-Backward Algorithm
  - The Intra-Frame Parallelism Algorithm

<p style="text-align: justify;">
 The static library files for windows (.lib) are provided for each algorithm and there are test files that demonstrate the usage of these libraries. The libraries were made using Visual Studio 2015 and the are available for both debug and release builds.

In the first algorithm, the PLL is applied to multiple frames simultaneously. To fully understand the details of this work and to see the results, please visit [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4100444) article.

In the second algorithm, as the name suggests, and in contrast to the first algorithm, parallelism is applied within one frame, meaning that multiple successive symbols are compensated in parallel.
</p>

## Signal

In this folder we upload some signals for testing the algorithms.

**For More Details You can contact me by: <ins>mh.shammakhi@gmail.com</ins>**
