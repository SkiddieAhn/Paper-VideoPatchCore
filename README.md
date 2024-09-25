# VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection (ACCV'24)
[![arXiv](https://img.shields.io/badge/arXiv-<2409.16225>-<COLOR>.svg)](https://arxiv.org/abs/2409.16225v1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-cuhk-avenue)](https://paperswithcode.com/sota/video-anomaly-detection-on-cuhk-avenue?p=videopatchcore-an-effective-method-to)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-iitb-corridor-1)](https://paperswithcode.com/sota/video-anomaly-detection-on-iitb-corridor-1?p=videopatchcore-an-effective-method-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-shanghaitech-4)](https://paperswithcode.com/sota/video-anomaly-detection-on-shanghaitech-4?p=videopatchcore-an-effective-method-to)

This repository is the ```official open-source``` of [VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection](https://arxiv.org/abs/2409.16225v1)
by Sunghyun Ahn, Youngwan Jo, Kijung Lee and Sanghyun Park.

## 📣 News
* **[2024/09/20]** Our **VPC** paper has been published in ACCV 2024!

## Description
Currently, VAD is gaining attention with memory techniques that store the features of normal frames. The stored features are utilized for frame reconstruction, identifying an abnormality when a significant difference exists between the
reconstructed and input frames. However, this approach faces several challenges due to the simultaneous optimization required for both the memory and encoder-decoder model. These challenges include increased optimization difficulty, complexity of implementation, and performance variability depending on the memory size. **To address these challenges, we propose an effective memory method for VAD, called VideoPatchCore. Inspired by PatchCore, our approach introduces a structure that prioritizes memory optimization and configures three types of memory tailored to the characteristics of video data.** This method effectively addresses the limitations of existing memory-based methods, achieving good performance comparable to state-of-the-art methods.  
<img width="750" alt="fig-vpc" src="https://github.com/user-attachments/assets/d8dda0a3-ebe3-4de0-96f9-ce5c764c949c">  
