import numpy as np
import cv2
from huffman import huffman_encode, huffman_decode

# DCVC-like compression: DCT + differencing (Section 5.2.1)
def compress_frame_dcvc(frame, quant_step, prev_frame=None):
    frame_float = frame.astype(np.float32)
    if prev_frame is not None:
        diff_frame = frame_float - prev_frame.astype(np.float32)
    else:
        diff_frame = frame_float
    dct_frame = cv2.dct(diff_frame)
    quant_frame = np.round(dct_frame / quant_step)
    return quant_frame, frame.shape, quant_step

# SlimVC-like compression: Variable quantization (Section 5.2.3)
def compress_frame_slimvc(frame, quant_step):
    return compress_frame_dcvc(frame, quant_step, None)

# Entropy-like compression: Huffman coding (Section 5.2.2)
def compress_frame_entropy(frame, quant_step, prev_frame=None):
    quant_frame, frame_shape, quant_step = compress_frame_dcvc(frame, quant_step, prev_frame)
    flat_quant = quant_frame.flatten().astype(np.int16)
    encoded, codebook = huffman_encode(flat_quant)
    return encoded, frame_shape, quant_step, codebook

# ROI-enhanced compression
def compress_frame_roi(frame, base_quant_step, prev_frame=None):
    # Edge detection for ROI (simple Canny)
    edges = cv2.Canny(frame, 100, 200)
    # Create ROI mask (1 for ROI, 0 for non-ROI)
    roi_mask = (edges > 0).astype(np.uint8)
    # Dilate mask to expand ROI slightly
    kernel = np.ones((5, 5), np.uint8)
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
    
    frame_float = frame.astype(np.float32)
    if prev_frame is not None:
        diff_frame = frame_float - prev_frame.astype(np.float32)
    else:
        diff_frame = frame_float
    dct_frame = cv2.dct(diff_frame)
    
    # Apply variable quantization: lower for ROI, higher for non-ROI
    quant_frame = np.zeros_like(dct_frame)
    quant_frame[roi_mask == 1] = np.round(dct_frame[roi_mask == 1] / (base_quant_step / 2))  # Higher quality
    quant_frame[roi_mask == 0] = np.round(dct_frame[roi_mask == 0] / (base_quant_step * 2))  # Lower quality
    
    return quant_frame, frame.shape, base_quant_step, roi_mask