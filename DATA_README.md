# Data Preperation
We provide cloud links for downloading the datasets used in this project, along with their directory structure and code.  
For the IITB Corridor dataset, due to its large size, we provide data preprocessing code instead.  
We express our sincere gratitude to researchers **Cewu Lu [[Avenue](https://openaccess.thecvf.com/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf)], 
Weixin Luo & Wen Liu [[SHTech](https://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf)], 
and Royston Rodrigues [[Corridor](https://openaccess.thecvf.com/content_WACV_2020/papers/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.pdf)]** for providing their datasets.

## 1. CUHK Avenue 
The dataset can be downloaded from the **official website [[Link](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)]**. 
However, there might be some differences in the directory structure compared to the dataset used in this project. 
You can set up our data directory environment using the download link provided below.

|  File Name |  Download   |   
|:-----------|:-----------|
|avenue.zip| [Google Drive](https://drive.google.com/file/d/1maU_c4SkCidJnq1P3QZD5QZB2BFCEEij/view?usp=sharing)     |

```
avenue
├── testing
    └── frames
        └── 01
            ├── 0000.jpg
            ├── ...
            └── 1438.jpg
        ├── 02
        ├── ...
        └── 21
└── training
    └── frames
        ├── 01
            ├── 0000.jpg
            ├── ...
            └── 1363.jpg
        ├── 02
        ├── ...
        └── 16
├── avenue.mat
```


## 2. ShanghaiTech Campus
The dataset can be downloaded from the **official website [[Link](https://svip-lab.github.io/dataset/campus_dataset.html)]**. 
However, there might be some differences in the directory structure compared to the dataset used in this project. You can set up our data directory environment using the download link provided below.

|  File Name   |  Download   |   
|:-----------|:-----------|  
|shanghai.vol1.egg| [Naver Mybox](http://naver.me/5eUIRzw3)     |
|shanghai.vol2.egg| [Naver Mybox](http://naver.me/5IS6Dgod)     |
|shanghai.vol3.egg| [Naver Mybox](http://naver.me/5r9f7drR)     |
|shanghai.vol4.egg| [Naver Mybox](http://naver.me/5xjU8nOY)     |
|shanghai.vol5.egg| [Naver Mybox](http://naver.me/5tJ9D4xS)     |
|shanghai.vol6.egg| [Naver Mybox](http://naver.me/GgW39uKE)     |
|shanghai.vol7.egg| [Naver Mybox](http://naver.me/IItIHfp8)     |
|shanghai.vol8.egg| [Google Drive](https://drive.google.com/file/d/138-N0UbkZMSeU3gg7jzPd41jmpxqtv5Y/view?usp=sharing)     |
```
shanghai
├── testing
    └── 01_0014
        ├── 000.jpg
        ├── ...
        └── 264.jpg
    ├── 01_0015
    ├── ...
    └── 12_0175
├── training
    └── 01_001
        ├── 0.jpg
        ├── ...
        └── 763.jpg
    ├── 01_002
    ├── ...
    └── 13_007
├── testframemask
    ├── 01_0014.npy
    ├── ...
    └── 12_0175.npy
```

## 3. IITB Corridor 
The dataset can be downloaded from the **official website [[Link](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/)]**. 
However, there might be some differences in the directory structure compared to the dataset used in this project.
Therefore, please restructure the directories according to the structure below before running the experiments.

```
iitb
├── testing
    └── 000209
        ├── 0000.jpg
        ├── ...
        └── 0547.jpg
    ├── 000210
    ├── ...
    └── 000358
├── training
    └── 000001
        ├── 0000.jpg
        ├── ...
        └── 0341.jpg
    ├── 000002
    ├── ...
    └── 000208
├── groundtruth
    └── 000209
        └── 000209.npy
    ├── 000210
    ├── ...
    └── 000358
```
We downloaded the dataset from the **official website** and executed the following code in the ``Test_IITB-Corridor`` directory. Below is the code we created to convert video into frames and save them.
Once the frame conversion is complete, you can restructure the directories according to the above structure. The training dataset has been processed in the same way.

```
import sys
sys.path.append('.')

import cv2
import os
import glob
import re
from fastprogress import progress_bar


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))

#==========(Test/Train)==========#
work='Test'
#================================#

video_list = glob.glob(f'./{work}/*')
video_list = sorted(video_list, key=extract_numbers)
print(video_list)  # ['./Test/000209', './Test/000210', './Test/000211', ...]

video_name_list = os.listdir(f'./{work}')
video_name_list = sorted(video_name_list, key=extract_numbers)
print(video_name_list) # ['000209', '000210', '000211', ...]

for i, video in progress_bar(enumerate(video_list), total=len(video_list)):
    video_name = video_name_list[i]
    video += f'/{video_name}.avi'

    print(f'--------------{i, video_name}------------------')
    if not os.path.exists(f"./{work}_fr/{video_name}"):
        os.makedirs(f"./{work}_fr/{video_name}")

    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(f"./{work}_fr/{video_name}/%04d.jpg" % count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      count += 1

    print("finish! convert video to frame")
```
