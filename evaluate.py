import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# Compute PSNR (Section 5.3)
def compute_psnr(original, reconstructed):
    return peak_signal_noise_ratio(original, reconstructed, data_range=255)

# Compute compression ratio (Section 5.3)
def compute_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size if compressed_size > 0 else 1