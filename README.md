# SegmentationUNet


# Segmentation with U-Net
This project implements a U-Net, a powerful convolutional neural network architecture, to perform semantic segmentation on images. The goal is to create a pixel-perfect mask that separates an animal from their background.


# Tech Stack:

Python

TensorFlow / Keras

TensorFlow Datasets (for easy data loading)

NumPy

Matplotlib

OpenCV-Python (for image manipulation)

# Architecture: U-Net

The U-Net model consists of an encoder (downsampler) and a decoder (upsampler) with "skip connections."

The encoder captures the context of the image.

The decoder uses the contextual information to reconstruct a pixel-level segmentation mask.

Skip connections are crucial as they feed information from the encoder directly to the decoder, helping it recover fine details (like strands of hair) for a precise outline.
