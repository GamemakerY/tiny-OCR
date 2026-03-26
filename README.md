# Tiny-OCR
TinyOCR: A proof-of-concept prototype of an OCR that I built as  a challenge.x

## Constraints

This was originally built under certain constraints mentioned below. All constraints were met:

**Model size:** All model weights ≤ 10 MB.

**No compression hacks:** Quantization, pruning, or external weight manipulation is
not allowed.

Achieved the model.bin file to be 1.6 MB (Without any compression hacks)

**Offline:** Fully offline inference. No internet.

**Hardware target:** Low-end devices (2 CPU cores, 2 GB RAM).

**Inference latency target:** ≤ 2 seconds per input.

**Encourage algorithmic preprocessing (e.g., edge detection, thresholding)
rather than heavy-weight ML models.**

## System Architecture
The OCR process is split into a two-stage pipeline.

### Phase 1: Image Preprocessing (OpenCV)

**Grayscale & Thresholding:** Converts the image to binary (black and white) to remove noise.

**Contour Detection:** Identifies individual character boundaries.

**Parsing:** Extracts and resizes each detected character to a uniform square input (28x28) for the neural network.

#### Phase 2: Inference (PyTorch)
The extracted characters are passed into a custom version of the ShuffleNet_V2:

**Model Optimization:** I utilized Transfer Learning on a pre-trained ShuffleNet_V2 architecture.

**Fine-Tuning:** By unfreezing the weights and re-training the final layers, I reduced the output classes to 47 (from 1000+), causing a huge reduction in size.

**Size Reduction:** The final model.bin is optimized to stay under 2 MB (1.6 MB), without any quantization, pruning or other compression hacks.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## How To Use

Simply input the relative file path or press enter to see a sample (Stored in images/)

There are a lot of features that could be way better, especially the accuracy.
