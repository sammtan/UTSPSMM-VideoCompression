import numpy as np
import cv2
from huffman import huffman_decode

# Decompress DCVC/SlimVC-like frame (Section 5.2.1, 5.2.3)
def decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_frame=None):
    dequant_frame = quant_frame * quant_step
    recon_diff = cv2.idct(dequant_frame.astype(np.float32))
    if prev_frame is not None:
        recon_frame = prev_frame.astype(np.float32) + recon_diff
    else:
        recon_frame = recon_diff
    return np.clip(recon_frame, 0, 255).astype(np.uint8)

# Decompress Entropy-like frame (Section 5.2.2)
def decompress_frame_entropy(encoded, frame_shape, quant_step, codebook, prev_frame=None):
    flat_quant = huffman_decode(encoded, codebook)
    quant_frame = flat_quant.reshape(frame_shape).astype(np.float32)
    return decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_frame)

# Decompress ROI-enhanced frame
def decompress_frame_roi(quant_frame, frame_shape, base_quant_step, roi_mask, prev_frame=None):
    # Reverse variable quantization
    dequant_frame = np.zeros_like(quant_frame, dtype=np.float32)
    dequant_frame[roi_mask == 1] = quant_frame[roi_mask == 1] * (base_quant_step / 2)  # ROI
    dequant_frame[roi_mask == 0] = quant_frame[roi_mask == 0] * (base_quant_step * 2)  # Non-ROI
    recon_diff = cv2.idct(dequant_frame)
    if prev_frame is not None:
        recon_frame = prev_frame.astype(np.float32) + recon_diff
    else:
        recon_frame = recon_diff
    return np.clip(recon_frame, 0, 255).astype(np.uint8)

# INR-like decompression (Section 5.2.4)
def decompress_frame_inr(key_frames, frame_idx, frame_shape):
    key_indices = sorted(key_frames.keys())
    if frame_idx in key_frames:
        return key_frames[frame_idx]
    for i in range(len(key_indices) - 1):
        if key_indices[i] < frame_idx < key_indices[i + 1]:
            t = (frame_idx - key_indices[i]) / (key_indices[i + 1] - key_indices[i])
            prev_frame = key_frames[key_indices[i]].astype(np.float32)
            next_frame = key_frames[key_indices[i + 1]].astype(np.float32)
            interp_frame = (1 - t) * prev_frame + t * next_frame
            return np.clip(interp_frame, 0, 255).astype(np.uint8)
    return key_frames[key_indices[0]]  # Fallback