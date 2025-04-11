import cv2
import numpy as np
from compress import compress_frame_dcvc, compress_frame_slimvc, compress_frame_entropy, compress_frame_roi
from decompress import decompress_frame_dcvc, decompress_frame_entropy, decompress_frame_roi, decompress_frame_inr
from evaluate import compute_psnr, compute_compression_ratio

def main():
    video_path = "ShakeNDry.mp4"  # Converted UVG-HD file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video. Ensure it’s a valid MP4 file.")
        return

    frames = []
    max_frames = 10
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)
        frame_count += 1
    cap.release()
    
    if not frames:
        print("Error: No frames loaded.")
        return

    quant_levels = [5, 20]  # SlimVC-like (Section 5.2.3)
    key_frame_interval = 5  # INR-like (Section 5.2.4)
    compressed_data = []
    key_frames = {}
    prev_frame = None

    # Compress frames
    for i, frame in enumerate(frames):
        quant_step = quant_levels[i % 2]
        if i % 4 == 0:  # DCVC
            compressed = compress_frame_dcvc(frame, quant_step, prev_frame)
        elif i % 4 == 1:  # Entropy
            compressed = compress_frame_entropy(frame, quant_step, prev_frame)
        elif i % 4 == 2:  # SlimVC
            compressed = compress_frame_slimvc(frame, quant_step)
        else:  # ROI
            compressed = compress_frame_roi(frame, quant_step, prev_frame)
        
        compressed_data.append(compressed)
        if i % key_frame_interval == 0:
            key_frames[i] = frame
        prev_frame = frame

    # Decompress frames
    recon_frames = []
    prev_recon = None
    for i, data in enumerate(compressed_data):
        if i % 4 == 0:  # DCVC
            quant_frame, frame_shape, quant_step = data
            recon_frame = decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_recon)
        elif i % 4 == 1:  # Entropy
            encoded, frame_shape, quant_step, codebook = data
            recon_frame = decompress_frame_entropy(encoded, frame_shape, quant_step, codebook, prev_recon)
        elif i % 4 == 2:  # SlimVC
            quant_frame, frame_shape, quant_step = data
            recon_frame = decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_recon)
        else:  # ROI
            quant_frame, frame_shape, quant_step, roi_mask = data
            recon_frame = decompress_frame_roi(quant_frame, frame_shape, quant_step, roi_mask, prev_recon)
        # INR override for non-key frames
        if i not in key_frames:
            recon_frame = decompress_frame_inr(key_frames, i, frames[0].shape)
        recon_frames.append(recon_frame)
        prev_recon = recon_frame

    # Evaluate (Section 5.3)
    psnr_values = []
    for orig, recon in zip(frames, recon_frames):
        psnr = compute_psnr(orig, recon)
        psnr_values.append(psnr)
    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")

    orig_size = sum(f.size * 8 for f in frames)
    comp_size = sum(len(c[0]) if isinstance(c[0], str) else c[0].size * 8 for c in compressed_data)
    ratio = orig_size / comp_size if comp_size > 0 else 1
    print(f"Compression Ratio: {ratio:.2f}:1")

if __name__ == "__main__":
    main()