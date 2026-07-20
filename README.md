# Software Defined Radio (SDR)
Created by Mohammad Hasan Shammakhi

This project contains:
1. [x] CPU-Based SDR
2. [x] GPU-Based SDR
3. [ ] FPGA-Based SDR

The project consists of multiple directories, each serving a distinct purpose. Here's a concise overview of the main directories in this repository.

## GPU_Based

<p align="justify">
This directory contains the full GPU-based SDR pipeline (frequency filtering, timing recovery, and PLL) built into a single executable, <code>sdr_gpu</code>. Pre-built binaries and object files are provided for several GPU architectures (<code>sm_61</code>, <code>sm_75</code>, <code>sm_86</code>, <code>sm_89</code>), along with the runtime <code>config.ini</code>, filter coefficient data in <code>data/</code>, and helper scripts in <code>scripts/</code> (e.g. <code>output_check.py</code> to compute EVM and plot the output constellation).
</p>

Use the root [Makefile](Makefile) to stage the right binary for your GPU:

```bash
make gpu-version   # detects your GPU's compute capability via nvidia-smi and
                    # stages the closest matching sdr_gpu build, config.ini,
                    # and data/ into build/
cd build
./sdr_gpu -i ./config.ini
python3 output_check.py
```

<p align="justify">
The executable requires the config file to be passed explicitly via <code>-i</code>. Before running, make sure the parameters in <code>config.ini</code> accurately match the signal you're processing, otherwise the output will be wrong:
</p>

- `input` — path to the input signal file (must match the actual signal you intend to process).
- `output` — path where the demodulated output will be written.
- `filterbb` — path to the baseband filter coefficients file.
- `sps`, `Rs`, `BW`, `central_freq`, `rollOff` — must match the true parameters of the input signal (samples per symbol, symbol rate, bandwidth, center frequency, and filter roll-off). A mismatch here (e.g. the wrong `sps`) will produce a corrupted or meaningless output.

## FrequencyFilter

<p align="justify">

In this directory, you'll find the source code for a frequency-domain filter. Applying the filter on a signal becomes highly efficient in the frequency domain, especially as the input signal length increases. Additionally, our implementation includes an optional upsampling by a factor of two, which was required for our SDR system with a minimum SPS of 4. The function utilizes the cuFFT library provided by CUDA.
</p>

## OtherBlocks

<p align="justify">
This directory contains commonly used blocks in SDR systems, excluding the demodulation section. Currently, it includes a digital downconverter (DDC) referred to as *Baseband* in the source code, and a time-domain filter. While the frequency domain version is available, the time domain version is simpler to implement and offers satisfactory performance in many applications.
</p>

## Signal

In this folder, we provide a selection of signals that can be used to test and evaluate the performance of the algorithms. These signals serve as representative examples to assess the effectiveness and accuracy of the implemented algorithms. By using these test signals, users can verify the functionality and suitability of the algorithms for their specific requirements.

**For more details or inquiries, feel free to reach out to me at <ins>mh.shammakhi@gmail.com</ins>. I am available to provide further information or answer any questions you may have.**
