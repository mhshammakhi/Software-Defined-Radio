# GPU-Based SDR

This project is GPU-Based SDR.
You can run this project to demodulate signals and export symbols. 
This project made in Visual Studio 2019.
It Contains:

- FrequencyFilter
- Resampler
- Matched Filter
- Power Normalizer
- Gardner SymbolSynchronizer
- Phased Locked Loop (CostasLoop)

**To test the SDR, you have the option to use signals provided in the "./signal" directory or utilize your own signal.**

Please ensure that the signal you provide follows the specified structure:

- The signal should be in a short (int16) format.
- It should contain complex numbers arranged in the following sequence:

re1 - im1 - re2 - im2 - re3 - im3 ..... - re(n) - im(n) - re(n+1) - im(n+1)

Here, re(n) represents the real part of the nth sample, and im(n) represents the imaginary part of the nth sample.

Feel free to use the provided signals or prepare your own signal for testing purposes.

**More details will be Added soon**
