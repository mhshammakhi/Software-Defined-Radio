#include "blocks.cuh"

__constant__ float c_bbFilterCoeffs[max_bbfilter_len];

//--- Downconverter
__global__
void Baseband(cuComplex inOut[], const float* freq_init,
    const int dataLength, const float frequency)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    float local_freq_init = *freq_init;
    cuComplex local_input;

    float freq, omega;
    while (i < dataLength)
    {
        float freq = float(i % 10000) * frequency;
        omega = 2.0 * M_PI * (freq - int(freq)) + local_freq_init;
        local_input = inOut[i];
        inOut[i].x = local_input.x * cosf(omega) + local_input.y * sinf(omega);
        inOut[i].y = local_input.y * cosf(omega) - local_input.x * sinf(omega);
        i += blockDim.x * gridDim.x;
    }
}

__global__
void Baseband_Update_State(float* freq_init, const float frequency, const int dataLength) {
    float freqLast = float(dataLength % 10000) * frequency;
    *freq_init += 2.0 * M_PI * (freqLast - int(freqLast));
}

//--- Baseband Filter
void setBBFilterCoeffsConstMem(const float cpu_array[], const int& filterLen) {
    assert(max_bbfilter_len >= filterLen);
    cudaMemcpyToSymbolAsync(c_bbFilterCoeffs, cpu_array, filterLen * sizeof(float), 0, cudaMemcpyHostToDevice);
}

__global__
void BasebandFilter(cuComplex output[], const cuComplex input[],
    const int filterLength, const int dataLength) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    cuComplex sum;
    while (i < dataLength)
    {
        sum = { 0.f, 0.f };

        for (int j = 0; j < filterLength; j++)
        {
            sum.x += c_bbFilterCoeffs[j] * input[i - j + filterLength - 1].x;
            sum.y += c_bbFilterCoeffs[j] * input[i - j + filterLength - 1].y;
        }

        output[i] = sum;
        i += (blockDim.x * gridDim.x);
    }
}

//--- Demappers

int getBitsPerSymbol(const ModulationType& mod_type) {
	int nBitsPerSymbol;

	switch (mod_type) {

	case ModulationType::BPSK:
		nBitsPerSymbol = 1;
		break;

	case ModulationType::QPSK:
		nBitsPerSymbol = 2;
		break;

	case ModulationType::_8PSK:
		nBitsPerSymbol = 3;
		break;

	case ModulationType::_16QAM:
		nBitsPerSymbol = 4;
		break;

	default:
		std::cout << "Modulation type is not supported\n";
		exit(1);

	}
	return nBitsPerSymbol;
}

void Demapper(const ModulationType& mod_type, uint8_t *outputBits, const cuComplex *in, const int dataLen) {
	switch (mod_type) {

	case ModulationType::BPSK:
		Demapper_BPSK << <12, 1024 >> > (outputBits, in, dataLen);
		break;

	case ModulationType::QPSK:
		Demapper_QPSK << <12, 1024 >> > (outputBits, in, dataLen);
		break;

	case ModulationType::_8PSK:
		Demapper_8PSK << <12, 1024 >> > (outputBits, in, dataLen);
		break;
		
	case ModulationType::_16QAM:
		Demapper_16QAM << <12, 1024 >> > (outputBits, in, dataLen);
		break;

	default:
		std::cout << "Modulation type is not supported\n";
		exit(1);

	}
}

__global__ void Demapper_BPSK(uint8_t *outputBits, const cuComplex *in, const int dataLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};

    while(i < dataLen){
        outputBits[i] = (in[i].x > 0 ? 1 : 0);
        i += (blockDim.x * gridDim.x);
    }
}

__global__ void Demapper_QPSK(uint8_t *outputBits, const cuComplex *in, const int dataLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};
    uint8_t temp;

    while(i < dataLen){
        temp = in[i].x > 0 ? (in[i].y > 0 ? 0 : 3) : (in[i].y > 0 ? 1 : 2);
        dec2bin(2, temp, &outputBits[i*2]);
        i += (blockDim.x * gridDim.x);
    }
}

__global__ void Demapper_8PSK(uint8_t *outputBits, const cuComplex *in, const int dataLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};

    cuComplex in_tmp;
    float val_re, val_im;
    uint8_t temp;
    
    while(i < dataLen){
        in_tmp = in[i];
        val_re = in_tmp.x * 0.923879532511287 - in_tmp.y * 0.382683432365090;
        val_im = in_tmp.x * 0.382683432365090 + in_tmp.y * 0.923879532511287;

        temp = val_re > 0 ? (val_im > 0 ? (val_re > val_im ? 0 : 1) : (val_re > -val_im ? 7 : 6)) : (val_im > 0 ? (-val_re > val_im ? 3 : 2) : (-val_re > -val_im ? 4 : 5));
        dec2bin(3, temp, &outputBits[i*3]);

        i += (blockDim.x * gridDim.x);
    }
}

__global__ void Demapper_16QAM(uint8_t *outputBits, const cuComplex *in, const int dataLen){
    int i{threadIdx.x + blockDim.x * blockIdx.x};

    cuComplex in_tmp;
    uint8_t temp;
    const float gain{1}, margin{2*gain};

    while(i < dataLen){
        in_tmp = in[i];
        
        if(in_tmp.x > 0){
            if(in_tmp.y > 0){
                temp = in_tmp.x > margin ? (in_tmp.y > margin ? 0 : 3) : (in_tmp.y > margin ? 1 : 2);
            }
            else{
                temp = in_tmp.x > margin ? (in_tmp.y > -margin ? 12 : 15) : (in_tmp.y > -margin ? 13 : 14);
            }
        }
        else{
            if(in_tmp.y > 0){
                temp = in_tmp.x > -margin ? (in_tmp.y > margin ? 4 : 7) : (in_tmp.y > margin ? 5 : 6);
            }
            else{
                temp = in_tmp.x > -margin ? (in_tmp.y > -margin ? 8 : 11) : (in_tmp.y > -margin ? 9 : 10);
            }
        }
        dec2bin(4, temp, &outputBits[i*4]);
        i += (blockDim.x * gridDim.x);
    }
}

__device__ __forceinline__ void dec2bin(int size, uint8_t z, uint8_t *z_bit){
    int i = 0;
    while (z > 0) {
        z_bit[size - 1 - i] = z % 2;
        z = z / 2;
        i++;
    }
    while(i < size){
        z_bit[size - 1 - i] = 0;
        i++;
    }
}