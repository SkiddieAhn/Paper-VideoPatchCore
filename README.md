# VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection (ACCV'24)

[![ArXiv](https://img.shields.io/badge/cs.CV-2409.16225-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2409.16225)
[![CvF](https://img.shields.io/badge/CvF-Website-DO9874)](https://openaccess.thecvf.com/content/ACCV2024/html/Ahn_VideoPatchCore_An_Effective_Method_to_Memorize_Normality_for_Video_Anomaly_ACCV_2024_paper.html)
[![Project](https://img.shields.io/badge/Project-Website-87CEEB)](https://shacoding.com/2024/10/23/videopatchcore-an-effective-method-to-memorize-normality-for-video-anomaly-detection-accv-2024/)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://shacoding.com/wp-content/uploads/2024/10/vpc_ms_thesis_sha.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-cuhk-avenue)](https://paperswithcode.com/sota/video-anomaly-detection-on-cuhk-avenue?p=videopatchcore-an-effective-method-to)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-iitb-corridor-1)](https://paperswithcode.com/sota/video-anomaly-detection-on-iitb-corridor-1?p=videopatchcore-an-effective-method-to)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/videopatchcore-an-effective-method-to/video-anomaly-detection-on-shanghaitech-4)](https://paperswithcode.com/sota/video-anomaly-detection-on-shanghaitech-4?p=videopatchcore-an-effective-method-to)

This repository is the ```official open-source``` of [VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection](https://openaccess.thecvf.com/content/ACCV2024/papers/Ahn_VideoPatchCore_An_Effective_Method_to_Memorize_Normality_for_Video_Anomaly_ACCV_2024_paper.pdf)
by Sunghyun Ahn, Youngwan Jo, Kijung Lee and Sanghyun Park.



## 📣 News
* **[2024/10/09]** **Instructions for data preparation** are released!
* **[2024/09/25]** Our **codes and memories** are released!
* **[2024/09/20]** Our VPC paper has been accepted to **ACCV 2024**!

## Description
Currently, VAD is gaining attention with memory techniques that store the features of normal frames. The stored features are utilized for frame reconstruction or prediction, identifying an abnormality when a significant difference exists between the generated and GT frames. However, this approach faces several challenges due to the simultaneous optimization required for both the memory and encoder-decoder model. These challenges include increased optimization difficulty, complexity of implementation, and performance variability depending on the memory size. **To address these challenges, we propose an effective memory method for VAD, called VideoPatchCore. Inspired by PatchCore, our approach introduces a structure that prioritizes memory optimization and configures three types of memory tailored to the characteristics of video data.** This method effectively addresses the limitations of existing memory-based methods, achieving good performance comparable to state-of-the-art methods.  
<img width="750" alt="fig-vpc" src="https://github.com/user-attachments/assets/35759d9d-aa4f-4965-9c59-8dce55a34cd5">  

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
- Please follow the [instructions](https://github.com/SkiddieAhn/Paper-VideoPatchCore/blob/main/DATA_README.md) to prepare the training and testing dataset.
- You can specify the dataset's path by editing ```'data_root'``` in ```extra/config.py```.
  
|     CUHK Avenue    | Shnaghai Tech.    |IITB Corridor    |
|:------------------------:|:-----------:|:-----------:|
|[Official Site](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)|[Official Site](https://svip-lab.github.io/dataset/campus_dataset.html)|[Official Site](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/)|

## Object Detection
- Input the ```path``` of the working directory where the  object files  will be stored in ```'work_dir```' of ```ObjectDetection/extra/config.py```
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

```Shell
# recommended option for avenue dataset (w/o pickle)
python run.py --work_num=0 --dataset=avenue --consecutive=10 
# recommended option for shanghai dataset (w/o pickle)
python run.py --work_num=0 --dataset=shanghai --consecutive=4 
# recommended option for iitb dataset (w/o pickle)
python run.py --work_num=0 --dataset=iitb --consecutive=4 
# recommended option for avenue dataset (w/ pickle)
python run.py --work_num=0 --dataset=avenue --consecutive=10 --is_load_test_pickle=True
# recommended option for shanghai dataset (w/ pickle)
python run.py --work_num=0 --dataset=shanghai --consecutive=4 --is_load_test_pickle=True
# recommended option for iitb dataset (w/ pickle)
python run.py --work_num=0 --dataset=iitb --consecutive=4 --is_load_test_pickle=True
```

## Memorization and Inference
- Input the ```path``` used for Object Detection in ```'work_dir```' of ```Memorization/extra/config.py```
- Navigate to the ```Memorization``` directory and enter the following command.
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
- We provide the **structure** of the working directory. Please refer to the link below.
- We provide the **code** used for the experiments through ```Google Colab```.

|     Working Directory    |  Directory Structure    |   Experiment Code    | 
|:------------------------:|:------------------------:|:------------------------:|
|[Google Drive](https://drive.google.com/file/d/1d3JZzlThsKq4qsuHnUTPrxJ4o8HWV50F/view?usp=drive_link)|[Github README](https://github.com/SkiddieAhn/Paper-VideoPatchCore/blob/main/WORK_README.md)|[Google Colab](https://colab.research.google.com/drive/1AuX7_f944_fcAA_4GPmutqqLMhOgBJUb?usp=sharing)|


## Qualitative Evaluation 
- We achieved excellent video anomaly detection by leveraging **three memory components** effectively.


|                       |Demo  |
|:--------------:|:-----------:|
| **Running** |![c1](https://github.com/user-attachments/assets/a7fa27aa-9053-4742-b73c-bc6bf46d23dd)|
| **Jumping** |![c2](https://github.com/user-attachments/assets/c244a8cb-a80f-4cb4-a72d-fe27f0a9e07c)|
| **Throwing a bag** |![c3](https://github.com/user-attachments/assets/36a7787f-7467-427a-b8db-f6a99d500f35)|
| **Wrong direction** |![c4](https://github.com/user-attachments/assets/5bea7bfe-ce3b-4eed-9628-c9345b65642c)|


## Citation
If you use our work, please consider citing:  
```Shell
@InProceedings{Ahn_2024_ACCV,
    author    = {Ahn, Sunghyun and Jo, Youngwan and Lee, Kijung and Park, Sanghyun},
    title     = {VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {2179-2195}
}
```

## Contact
Should you have any question, please create an issue on this repository or contact me at skd@yonsei.ac.kr.
