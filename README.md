# Dynamic Incremental Regularised Adaptation - Self Supervised (DIRA-SS)

This is the official repository for our paper: [*DIRA-SS: Dynamic Incremental Regularised Adaptation (Self Supervised)*](https://arxiv.org/abs/2311.07461v2)

Autonomous systems (AS) often use Deep Neural Network (DNN) classifiers to enable their operation in high-dimensional, non-linear, and dynamically evolving environments. Given the complexity and scale of these environments, DNN classifiers are prone to encountering distributional shifts during operation. Consequently, they may produce misclassifications when faced with domains not recognised during initial training. To enhance AS reliability and address this limitation, DNN classifiers must be capable of adapting during operation when encountering various operational domains using a limited number of samples (e.g., 2 to 100 samples). In this paper, we introduce an approach for self-supervised dynamic incremental regularised adaptation (DIRA-SS). Our work extends DIRA, a supervised method for domain adaption, that was previously shown to perform competitively with state-of-the-art methods in domain adaptation. DIRA-SS advances DIRA, enabling its functioning in unsupervised settings, where labels cannot be provided. Our approach is evaluated on different image classification benchmarks aimed at evaluating robustness to distribution shifts (e.g.CIFAR-10C/100C, ImageNet-C), and produces competitive state-of-the-art performance in comparison with other methods from the literature including DIRA.

# Installation
1) Clone this repository: `git clone` 
2) Clone our docker image and setup container: `docker run -t --runtime=nvidia --shm-size 16G -d --name netzip -v ~/gits:/home -p 5000:80 abanoubg/netzip:latest`.


# Setup Datasets

# Quick Start
