# RePixel

RePixel is a Python tool designed to try to reverse-engineer and restore the original resolution of upscaled and lossy compressed pixel art.

RePixel uses FFT and gradient analysis to automatically detect the pixel grid and reconstruct the original image.

## Features

- **Automatic Grid Detection:** Uses Fast Fourier Transform (FFT) to determine the underlying block size (scaling factor) without manual input.
- **Phase Alignment:** Automatically calculates grid offsets (X and Y), allowing it to work even if the image has been cropped or doesn't start perfectly at (0,0).
- **Smart Sampling:**
  - **Center Sampling:** Picks the pixel from the exact center of the block for crisp restoration.
  - **Average Sampling:** Option to average the colors within a block, useful if the source image has compression artifacts or noise.
- **Noise Robustness:** Optional Gaussian blur pre-processing to handle JPEG artifacts or noisy scans before grid detection.

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- SciPy

You can install the dependencies via pip:

```bash
pip install opencv-python numpy scipy
```

## Usage

### Basic Usage

The simplest way to use RePixel is to provide the input image. The script will automatically detect the grid and save the result as `filename_restored.png`.

```bash
python re_pixel.py image.png
```

### Advanced Usage

You can customize the output path, noise handling, and sampling method.

```bash
python re_pixel.py input.png -o output.png --blur 0.5 --radius 1
```

### Arguments

| Argument   | Flag             | Description                                                                                                                               |
| ---------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**  | `input`          | Path to the source image (Required).                                                                                                      |
| **Output** | `-o`, `--output` | Path to save the restored image. Defaults to `<name>_restored.png`.                                                                       |
| **Blur**   | `-b`, `--blur`   | Pre-processing Gaussian blur sigma. Increase this if the tool struggles to find the grid due to noise/JPEG artifacts. Default is `0`.     |
| **Radius** | `-r`, `--radius` | Sampling radius. `0` = Center pixel (Nearest Neighbor logic). `>0` = Averages a radius around the center (e.g., `1` averages a 3x3 area). |
| **Scale**  | `-s`, `--scale`  | Scale factor to enlarge the output image (e.g. 2, 4). Default: 1 (Original Size)                                                          |

## How It Works

1. The script calculates the pixel-to-pixel difference (gradients) along the X and Y axes to find "edges."
2. These gradients are summed up into 1D profiles (vertical and horizontal).
3. A Fast Fourier Transform is applied to the profiles to find the dominant frequency, which corresponds to the size of the "big" pixels (the block size).
4. The script slides a grid over the image to find the offset (phase) where the grid lines align perfectly with the highest gradient edges.
5. It iterates through the calculated grid, sampling the center of each block to rebuild the original low-resolution sprite.

## License

This project is open-source and available under the MIT License.

