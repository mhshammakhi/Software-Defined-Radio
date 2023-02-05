# Blocks Commonly Used In SDR Systems
In this sub-directory, the source codes of several common SDR blocks such as downconverter and baseband filter are provided. Examples on how to use these blocks are also available in test.cu file.

The codes are tested on CUDA 10.1 for Windows (MSVC). The block and grid sizes for kernel calls are optimized for GeForce GTX 1050Ti according to its specs such as max resident blocks and threads per SM. Make sure to set them properly for acheiving the best possible performance. 

## Contents

[Baseband](#baseband)

[Baseband Filter](#baseband-filter)


## Baseband
This block is actually the first step of a digital downconverter (DDC), that shifts the signal from its carrier frequency down to baseband. The block is simply a multiplication. A test function called test_basband() shows how to use this block. It performs this block on several frames of data and writes the output to a .bin file.

## Baseband Filter
This block is the implementation of a baseband filter of arbitrary length in time domain. The test function test_bbfilter() is an example of how to use this block. It performs this block on several frames of data and writes the output to a .bin file.