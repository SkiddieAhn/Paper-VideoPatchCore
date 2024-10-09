# Working Directory Structure

* ```shanghai_test.pickle```: file containing detected bounding boxes for the test dataset
*  ```spatial_memory_bank_0.25.pt```: file containing stored spatial patches (subsampling ratio: 0.25)
-------------------
* ```work_num```: 0
* ```cnl_pool```: 32 (avenue), 64 (shanghai, iitb)
* ```spatial_f_coreset```: 0.01 (avenue), 0.25 (shanghai), 0.10 (iitb)
* ```temporal_f_coreset```: 0.01 (avenue), 0.25 (shanghai), 0.10 (iitb)
* ```highlevel_f_coreset```: 0.01 (avenue), 0.25 (shanghai), 0.10 (iitb)
-------------------
* **object files** will be saved in the ```objects/{dataset}/{test or train}``` directory.
* **lf files** will be saved in the ```l_features/{dataset}/{cnl_pool}/visual/{test or train}``` directory.
* **gf files** will be saved in the ```g_features/{dataset}/{cnl_pool}/visual/{test or train}``` directory.

```Shell
0
    ├─g_features
    │  ├─avenue
    │  │  └─32
    │  │      │  highlevel_memory_bank_0.01.pt
    │  │      │
    │  │      └─visual
    │  │          ├─test
    │  │          └─train
    │  ├─iitb
    │  │  └─64
    │  │      │  highlevel_memory_bank_0.1.pt
    │  │      │
    │  │      └─visual
    │  │          ├─test
    │  │          └─train
    │  └─shanghai
    │      └─64
    │          │  highlevel_memory_bank_0.25.pt
    │          │
    │          └─visual
    │              ├─test
    │              └─train
    ├─l_features
    │  ├─avenue
    │  │  └─32
    │  │      │  spatial_memory_bank_0.01.pt
    │  │      │  temporal_memory_bank_0.01.pt
    │  │      │
    │  │      └─visual
    │  │          ├─test
    │  │          └─train
    │  ├─iitb
    │  │  └─64
    │  │      │  spatial_memory_bank_0.1.pt
    │  │      │  temporal_memory_bank_0.1.pt
    │  │      │
    │  │      └─visual
    │  │          ├─test
    │  │          └─train
    │  └─shanghai
    │      └─64
    │          │  spatial_memory_bank_0.25.pt
    │          │  temporal_memory_bank_0.25.pt
    │          │
    │          └─visual
    │              ├─test
    │              └─train
    └─objects
        ├─avenue
        │  │  avenue_test.pickle
        │  │  avenue_train.pickle
        │  │
        │  ├─test
        │  └─train
        ├─iitb
        │  │  iitb_test.pickle
        │  │  iitb_train.pickle
        │  │
        │  ├─test
        │  └─train
        └─shanghai
            │  shanghai_test.pickle
            │  shanghai_train.pickle
            │
            ├─test
            └─train
```
