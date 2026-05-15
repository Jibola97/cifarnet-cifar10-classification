# CIFAR-10 Classification with a Custom Intermediate-Block CNN

This repository contains a custom neural network for CIFAR-10 image classification.

The project implements a CNN architecture built around **intermediate blocks**, where multiple independent convolutional branches process the same input and are combined using weights generated from channel-wise averages. The final model achieved **95.04% test accuracy** on CIFAR-10.

## Project Overview

The aim of this project was to implement and improve a neural network architecture for CIFAR-10 classification while preserving a specific custom block mechanism.

CIFAR-10 contains 60,000 RGB images across 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The dataset is split into 50,000 training images and 10,000 test images.

## Architecture

The model is based on a sequence of custom **IntermediateBlocks**, followed by an output classification block.

Each intermediate block contains several independent convolutional branches. Instead of chaining these branches, each branch receives the same input tensor. The input is globally averaged across spatial dimensions, passed through a fully connected layer, and converted into mixing weights using softmax. These weights are then used to combine the outputs of the convolutional branches.

In simplified form:

Input x
   ↓
Parallel convolutional branches: C1(x), C2(x), ..., CL(x)
   ↓
Channel average of x → FC layer → softmax weights a1, a2, ..., aL
   ↓
Weighted output = a1C1(x) + a2C2(x) + ... + aLCL(x)

## Model Structure

| Block |  Channels | Branches | Stride | Spatial size  |
| ----- | --------: | -------: | -----: | ------------- |
| B1    |    3 → 64 |        2 |      1 | 32×32 → 32×32 |
| B2    |  64 → 128 |        3 |      2 | 32×32 → 16×16 |
| B3    | 128 → 128 |        3 |      1 | 16×16 → 16×16 |
| B4    | 128 → 256 |        4 |      2 | 16×16 → 8×8   |
| B5    | 256 → 256 |        4 |      1 | 8×8 → 8×8     |
| B6    | 256 → 512 |        4 |      2 | 8×8 → 4×4     |

## Training Setup

| Setting           |                                 Value |
| ----------------- | ------------------------------------: |
| Epochs            |                                   200 |
| Batch size        |                                   128 |
| Optimiser         |            SGD with Nesterov momentum |
| Learning rate     |                                   0.1 |
| Schedule          |      Linear warmup + cosine annealing |
| Weight decay      |                                  5e-4 |
| Loss              | CrossEntropyLoss with label smoothing |
| Dropout           |                                   0.4 |
| Gradient clipping |                        max_norm = 5.0 |


## Results 

Best test accuracy: 95.04%
Best epoch: 194

Per-class accuracy:

Class	Accuracy
airplane	96.0%
automobile	97.6%
bird	92.9%
cat	89.3%
deer	96.4%
dog	91.5%
frog	97.4%
horse	96.6%
ship	96.3%
truck	96.5%

The weakest classes were cat and dog, which is expected because they are visually similar at CIFAR-10’s 32×32 resolution. The largest confusion-matrix errors were between these two classes.

## Key Findings

The main findings from the experiment were:

The custom intermediate-block mechanism can achieve strong CIFAR-10 performance when combined with modern optimisation techniques.
BatchNorm, residual connections and a warmup/cosine learning-rate schedule were important for stable training.
Data augmentation and label smoothing helped reduce overfitting.
The final model reached over 95% test accuracy while preserving the required custom block structure.




