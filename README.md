# Demystifying ConvNets

Th goal is to maintain a collection of ipython notebooks to provide a less mathematical and more intuitive feel as to
 why convolutional nets work.

There are several excellent blog posts explaining what convolutions are, how Conv, ReLU, MaxPooling operations 
transform the image. Soon afterwards we learn that a common design choice is to stack layers as 
[[Conv -> Relu] * N] -> Pool] * M (ignoring batch norm to keep the discussion simple). Why should it be this way?

When I started learning about deep learning, I had a lot of questions on how they are applied in practice. For example:

- When is it a good idea to use MaxPooling vs average pooling? 
- How to handle large image sizes, say 1000 X 1000?
- What if the number of training examples are low? 
- Can we encode problem specific characteristics into convnet design? In medical image classification, 
the region of interest (ROI) is usually very small relative to the image. Can we design a conv net to somehow capture 
this knowledge a-priori?

This repo is organized into notebooks. Each notebook is designed to provide a rough intuition on various aspects of 
designing conv nets. We will skip math and jump jump right into the code.


# Notebooks

1. [Why should I use ReLU - A visual perspective](notebooks/exploring-convolutions.ipynb)

More to come soon...


# Contributing

The goal is to democratize AI and make it easier for newcomers entering the field.
If you have a unique perspective on some aspect of conv nets, please submit a PR.  
