# VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection (ACCV'24)
[![arXiv](https://img.shields.io/badge/arXiv-<2409.16225>-<COLOR>.svg)](https://arxiv.org/abs/2409.16225)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-cuhk-avenue)](https://paperswithcode.com/sota/video-anomaly-detection-on-cuhk-avenue?p=videopatchcore-an-effective-method-to)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-iitb-corridor-1)](https://paperswithcode.com/sota/video-anomaly-detection-on-iitb-corridor-1?p=videopatchcore-an-effective-method-to)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-shanghaitech-4)](https://paperswithcode.com/sota/video-anomaly-detection-on-shanghaitech-4?p=videopatchcore-an-effective-method-to)

This repository is the ```official open-source``` of [VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection](https://arxiv.org/pdf/2409.16225)
by Sunghyun Ahn, Youngwan Jo, Kijung Lee and Sanghyun Park.

## 📣 News
* **[2024/09/25]** **VPC** codes & memories are released!
* **[2024/09/20]** Our **VPC** paper has been accepted to ACCV 2024!

## Description
Currently, VAD is gaining attention with memory techniques that store the features of normal frames. The stored features are utilized for frame reconstruction or prediction, identifying an abnormality when a significant difference exists between the generated and GT frames. However, this approach faces several challenges due to the simultaneous optimization required for both the memory and encoder-decoder model. These challenges include increased optimization difficulty, complexity of implementation, and performance variability depending on the memory size. **To address these challenges, we propose an effective memory method for VAD, called VideoPatchCore. Inspired by PatchCore, our approach introduces a structure that prioritizes memory optimization and configures three types of memory tailored to the characteristics of video data.** This method effectively addresses the limitations of existing memory-based methods, achieving good performance comparable to state-of-the-art methods.  
<img width="750" alt="fig-vpc" src="https://github.com/user-attachments/assets/d8dda0a3-ebe3-4de0-96f9-ce5c764c949c">  

## Dependencies
- python >= 3.8  
- torch = 1.13.1+cu117
- torchvision = 0.14.1+cu117
- scikit-learn = 1.0.2
- opencv-python  
- h5py  
- fastprogress
- Other common packages.

## Notes
Vision Encoder is based on [openai-clip](https://github.com/openai/CLIP) and Object Detector is based on [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/). Please click the link to download the package. Thanks to the authors for their great work. 

## Datasets
- First, input the **path** of the **working directory** where the **(objects, features, and memories)** files will be stored in ```'work_dir```' of ```extra/config.py```.
- You can specify the dataset's path by editing ```'data_root'``` in ```extra/config.py```.
  
|     CUHK Avenue    | Shnaghai Tech.    |IITB Corridor    |
|:------------------------:|:-----------:|:-----------:|
|[Official Site](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)|[Official Site](https://svip-lab.github.io/dataset/campus_dataset.html)|[Official Site](https://github.com/Rodrigues-Royston/Multi-timescale_Trajectory_Prediction)|

## Object Detection
- Navigate to the ```ObjectDetection``` directory and enter the following command.
- You can input ```dataset_name``` as one of the following choices: **avenue**, **shanghai**, **iitb**.
- We set the ```consecutive``` to **10** for avenue and **4** for shanghai and iitb.
- **object files** are saved in the ```objects``` directory of the working directory.
```Shell
# default option for object detection
python run.py --work_num=0 --dataset={dataset_name}
# change number of input frames
python run.py --work_num=0 --dataset={dataset_name} --consecutive=10
# save bounding box file
python run.py --work_num=0 --dataset={dataset_name} --is_save_train_pickle=True --is_save_test_pickle=True
# load bounding box and save object batches
python run.py --work_num=0 --dataset={dataset_name} --is_load_train_pickle=True --is_load_test_pickle=True
# save images with bounding boxes
python run.py --work_num=0 --dataset={dataset_name} --save_image=True
# save all detected object images
python run.py --work_num=0 --dataset={dataset_name} --save_image_all=True 
```

## Memorization and Inference
- Navigate to the ```Memorization``` directory and enter the following command.
- Enter the following command to perform memorization and inference.
- **lf files** and **spatial & temporal memory banks** are saved in the ```l_features``` directory of the working directory.
- **gf files** and **high-level semantic memory bank** are saved in the ```g_features``` directory of the working directory.
```Shell
# recommended option for avenue dataset 
python run.py \
    --work_num=0 --consecutive=10 --dataset=avenue --cnl_pool=32 \
    --spatial_f_coreset=0.01 --temporal_f_coreset=0.01 --highlevel_f_coreset=0.01 

# recommended option for shanghai dataset 
python run.py \
    --work_num=0 --consecutive=4 --dataset=shanghai --cnl_pool=64 \
    --spatial_f_coreset=0.25 --temporal_f_coreset=0.25 --highlevel_f_coreset=0.25

# recommended option for iitb dataset 
python run.py \
    --work_num=0 --consecutive=4 --dataset=iitb --cnl_pool=64 \
    --spatial_f_coreset=0.1 --temporal_f_coreset=0.1 --highlevel_f_coreset=0.1

# save test features and perform inference using saved memories
python run.py \
    --work_num=0 --consecutive=4 --dataset=iitb --cnl_pool=64 \
    --spatial_f_coreset=0.1 --temporal_f_coreset=0.1 --highlevel_f_coreset=0.1 \
    --save_memory=False

# perform inference using the saved test features and memories
python run.py \
    --work_num=0 --consecutive=4 --dataset=iitb --cnl_pool=64 \
    --spatial_f_coreset=0.1 --temporal_f_coreset=0.1 --highlevel_f_coreset=0.1 \
    --save_feature=False --save_memory=False
```

## Bounding box and Memory
- The following is the working directory used for the experiments. This directory includes the **bounding box** and **memory** files.
- We provide the **code** used for the experiments through ```Google Colab```.

|     Working Directory    |  Experiment Code    | 
|:------------------------:|:------------------------:|
|[Google Drive](https://drive.google.com/file/d/1d3JZzlThsKq4qsuHnUTPrxJ4o8HWV50F/view?usp=drive_link)|[Google Colab](https://colab.research.google.com/drive/1AuX7_f944_fcAA_4GPmutqqLMhOgBJUb?usp=sharing)|


## Citation
If you use our work, please consider citing:  
```Shell
@misc{ahn2024videopatchcoreeffectivemethodmemorize,
      title={VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection},
      author={Sunghyun Ahn and Youngwan Jo and Kijung Lee and Sanghyun Park},
      year={2024},
      eprint={2409.16225},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.16225},
}
```

## Contact
Should you have any question, please create an issue on this repository or contact me at skd@yonsei.ac.kr.
