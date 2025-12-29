import argparse
import os

import cv2
import numpy as np
from scipy.signal import find_peaks


def restore_pixel_art(image_path, output_path, blur_sigma=0, sample_radius=0):
    """
    Restores the original resolution of upscaled pixel art by detecting the grid pattern.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the restored image.
        blur_sigma: Sigma for Gaussian blur during grid detection (0 = auto/off).
                    Useful for noisy images (e.g., JPEG artifacts).
        sample_radius: Radius for color sampling averaging.
                       0 = Center pixel only (nearest neighbor feel).
                       >0 = Average within the radius (smoother).
    """

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    height, width, _ = img.shape
    print(f"Original Image Size: {width}x{height}")

    # Step 1: Compute Color Gradients (Edge Detection)
    # Cast to float to prevent uint8 overflow during subtraction
    img_float = img.astype(float)

    # [Preprocessing] Apply Gaussian blur if specified
    # If sigma > 0, use it; otherwise skip or use minimal default logic if needed later
    if blur_sigma > 0:
        sigma = blur_sigma
        # Ensure kernel size is odd
        base_k = int(sigma * 3)
        ksize = base_k + 1 if base_k % 2 == 0 else base_k
        ksize = max(3, ksize)

        print(f"Preprocessing Blur: Sigma={sigma}, Kernel={ksize}x{ksize}")
        img_float = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

    # 2. Compute gradients along X-axis (diff between adjacent columns)
    diff_x = np.abs(img_float[:, 1:, :] - img_float[:, :-1, :])
    grad_x = np.sum(diff_x, axis=2)

    # 3. Compute gradients along Y-axis (diff between adjacent rows)
    diff_y = np.abs(img_float[1:, :, :] - img_float[:-1, :, :])
    grad_y = np.sum(diff_y, axis=2)

    # Project gradients to 1D profiles by summing along the perpendicular axis
    proj_x = np.sum(grad_x, axis=0)
    proj_y = np.sum(grad_y, axis=1)

    # Step 2: Detect Pixel Block Size using FFT
    def get_block_size(profile):
        n = len(profile)
        yf = np.fft.rfft(profile)
        amplitude = np.abs(yf)

        # Dynamically filter low-frequency noise (DC component and slow variations)
        max_possible_block_size = 64
        min_freq_index = int(n / max_possible_block_size)
        cutoff = max(min_freq_index, 5)
        amplitude[:cutoff] = 0

        # Find peaks in the frequency domain
        peaks, _ = find_peaks(amplitude, height=np.max(amplitude) * 0.25, distance=5)

        if len(peaks) == 0:
            return 1

        # The dominant peak represents the fundamental frequency of the grid
        dominant_peak_idx = peaks[np.argmax(amplitude[peaks])]
        if dominant_peak_idx == 0:
            return 1

        estimated_size = n / dominant_peak_idx
        block_size = int(round(estimated_size))
        return block_size if block_size > 1 else 1

    block_w = get_block_size(proj_x)
    block_h = get_block_size(proj_y)

    print(f"Estimated Block Size: Width={block_w}, Height={block_h}")

    if block_w <= 1 and block_h <= 1:
        print("Block size detected as 1. Skipping restoration.")
        return

    # Step 3: Determine Grid Phase & Offset

    def find_best_phase(gradient_profile, block_size):
        """
        Finds the phase of the grid lines.
        The phase corresponds to the index where the gradient accumulation is maximized
        (i.e., the edges of the pixel blocks).
        """
        if block_size <= 1:
            return 0

        best_phase = 0
        max_score = -1

        # Iterate through all possible phases (0 to block_size-1)
        for phase in range(block_size):
            # Extract gradient values at this phase interval
            indices = range(phase, len(gradient_profile), block_size)
            valid_indices = [i for i in indices if i < len(gradient_profile)]

            if not valid_indices:
                continue

            score = np.sum(gradient_profile[valid_indices])

            if score > max_score:
                max_score = score
                best_phase = phase

        return best_phase

    # Calculate optimal phase (location of grid lines)
    phase_x = find_best_phase(proj_x, block_w)
    phase_y = find_best_phase(proj_y, block_h)

    # [Offset Calculation]
    # The gradient peaks at the block boundary.
    # If the boundary is at index K, the current block ends at K, and the next starts at K+1.
    # Example: Block=10.
    # Case A: Offset=0. Pixels 0-9 are a block. Boundary at 9. Phase=9. Offset = (9+1)%10 = 0.
    # Case B: Offset=1. Pixel 0 is partial/junk. Pixels 1-10 are a block. Boundary at 0. Phase=0. Offset = (0+1)%10 = 1.
    offset_x = (phase_x + 1) % block_w
    offset_y = (phase_y + 1) % block_h

    print(
        f"Best Grid Offset: X={offset_x}, Y={offset_y} (Phase X={phase_x}, Phase Y={phase_y})"
    )

    # Step 4: Reconstruct Image via Grid Sampling

    new_w = (width - offset_x) // block_w
    new_h = (height - offset_y) // block_h

    # [Sampling Strategy]
    # If sample_radius > 0, apply a box blur first.
    # Sampling the center of a blurred image is mathematically equivalent to taking the average
    # of the neighborhood in the original image, but much faster (vectorized).
    sample_img = img
    if sample_radius > 0:
        # Kernel size must be odd: radius r -> diameter 2r+1
        avg_ksize = sample_radius * 2 + 1
        print(
            f"Applying Average Sampling with radius {sample_radius} (Kernel {avg_ksize}x{avg_ksize})"
        )
        sample_img = cv2.blur(img, (avg_ksize, avg_ksize))

    # Calculate center coordinates for every block
    y_coords = np.arange(offset_y + block_h // 2, height, block_h)[:new_h]
    x_coords = np.arange(offset_x + block_w // 2, width, block_w)[:new_w]

    # Generate grid of indices
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Clip coordinates to ensure they remain within image bounds
    yy = np.clip(yy, 0, height - 1)
    xx = np.clip(xx, 0, width - 1)

    # Vectorized sampling
    restored_img = sample_img[yy, xx]

    cv2.imwrite(output_path, restored_img)
    print(f"Restored image saved to {output_path} ({new_w}x{new_h})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Art Restoration Tool")

    # Positional arguments
    parser.add_argument("input", help="Path to input image file")

    # Optional arguments
    parser.add_argument(
        "-o", "--output", help="Path to output image file (default: <input>_restored)"
    )
    parser.add_argument(
        "-b",
        "--blur",
        type=float,
        default=0,
        help="Gaussian blur sigma for grid detection (default: 0)",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=0,
        help="Sampling radius for color averaging. 0=Center pixel only, 1=3x3 average... (default: 0)",
    )

    args = parser.parse_args()

    # Handle output path generation
    input_path = args.input
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_restored.png"

    restore_pixel_art(
        input_path, output_path, blur_sigma=args.blur, sample_radius=args.radius
    )
