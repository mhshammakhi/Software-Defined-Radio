import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Read signal
signal_path = os.path.join(os.path.dirname(__file__), './', 'SDR_Output.bin')
raw = np.fromfile(signal_path, dtype=np.float32, count=10_000_000)
output_signal = raw[0::2] + 1j * raw[1::2]

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--delay', type=int, default=1000)
args = parser.parse_args()

# Common parameters
plot_delay = args.delay

rx = output_signal[plot_delay:]
rx = rx / np.sqrt(np.mean(np.abs(rx) ** 2))

ref_const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
dists = np.abs(rx[:, None] - ref_const[None, :])
idx = np.argmin(dists, axis=1)
ref = ref_const[idx]

evm_rms = np.sqrt(np.mean(np.abs(rx - ref) ** 2))
evm_percent = evm_rms * 100
evm_db = 20 * np.log10(evm_rms)
print(f"EVM = {evm_percent:.2f}% or {evm_db:.2f} dB (PLL Output)")

# Scatter plot
plt.figure()
plt.scatter(rx.real, rx.imag, s=0.1, alpha=0.3)
plt.title('PLL Output')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()


