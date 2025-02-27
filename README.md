# Neha
# Enhanced UNet for Segmentation and Pseudo-Colorization of Satellite Images

## Overview

This project implements an Enhanced UNet model for the pseudo-colorization and Segmentation  of grayscale satellite images. The model takes grayscale  images as input and predicts colorized versions using a combination of convolutional layers with 3x3 and 5x5 filters, followed by a UNet architecture for feature extraction and reconstruction.

## Dataset

The dataset consists of satellite images stored in two folders:
- **Grayscale Images**: Located at `E:\Data\WaterBodiesDataset\GreyScaleImage`
- **Color Images**: Located at `E:\Data\WaterBodiesDataset\ColuredImages`

You can access the dataset from Kaggle: [Water Bodies Dataset](https://www.kaggle.com/)

## Features

- **Image Preprocessing**:
  - Resizes images to 256x256 for better resolution.
  - Applies histogram equalization to grayscale images.
  - Converts grayscale images to RGB format.
  - Normalizes images by scaling pixel values to the range [0, 1].
- **Enhanced UNet Architecture**:
  - Uses both 3x3 and 5x5 convolutional filters for feature extraction.
  - Encoder-decoder structure with batch normalization and ReLU activations.
  - Skip connections for preserving spatial information.
- **Training**:
  - Splits the dataset into 80% training and 20% validation.
  - Compiles the model using Adam optimizer and MSE loss.
  - Trains for 100 epochs with a batch size of 16.
- **Visualization**:
  - Displays grayscale, pseudo-colorized, and original color images side by side.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python matplotlib numpy
   ```

## Usage

1. Ensure your dataset is correctly placed:
   - Grayscale images: `E:\Data\WaterBodiesDataset\GreyScaleImage`
   - Color images: `E:\Data\WaterBodiesDataset\ColuredImages`
2. Run the script:
   ```bash
   python unet_colorization.py
   ```
3. The model will train and plot the results after completion.

## File Structure
```
.
├── unet_colorization.py  # Main script for model training and testing
├── README.md             # Project description
├── requirements.txt      # List of dependencies
└── E:\Data\WaterBodiesDataset
    ├── GreyScaleImage    # Grayscale images
    └── ColuredImages     # Corresponding color images
```

## Model Architecture

The Enhanced UNet model contains the following layers:
- **Input Layer**: Takes RGB grayscale images (converted to 3 channels).
- **Dual Convolution**: Applies both 3x3 and 5x5 convolutions.
- **Encoder**: Downsampling using Conv2D and MaxPooling layers.
- **Bottleneck**: Highest feature extraction layer.
- **Decoder**: Upsampling with concatenation to preserve spatial context.
- **Output Layer**: Produces a 3-channel RGB image using a 1x1 convolution with a sigmoid activation.

## Results

The script will plot the following after model training:
1. Grayscale Image
2. Predicted Pseudo-Colorized Image
3. Original Color Image

## Acknowledgments

- TensorFlow for deep learning.
- OpenCV for image preprocessing.
- Matplotlib for visualization.
This research work is part of the project supported by the Department of Science and Technology (DST) under the PURSE 2022 scheme at BIT Mesra and the grant number is SR/PURSE/2022/130(G)


Feel free to modify the paths or model architecture to suit your dataset and research goals!

