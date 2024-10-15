# Multitask-Optimization-Recommendation-Library
MTOReclib provides a PyTorch implementation of multi-task recommendation models and multi-task optimization methods.

<div align="left">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.12740-da282a.svg)](https://arxiv.org/abs/2410.05806)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center"> 
    <img src="https://github.com/yjdy/Multitask-Optimization-Recommendation-Library/blob/main/misc/toy.png" width="800">
</p>

## Datasets
* AliExpressDataset: This is a dataset gathered from real-world traffic logs of the search system in AliExpress. This dataset is collected from 5 countries: Russia, Spain, French, Netherlands, and America, which can utilized as 5 multi-task datasets. [Original_dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690) [Processed_dataset Google Drive](https://drive.google.com/drive/folders/1F0TqvMJvv-2pIeOKUw9deEtUxyYqXK6Y?usp=sharing) [Processed_dataset Baidu Netdisk](https://pan.baidu.com/s/1AfXoJSshjW-PILXZ6O19FA?pwd=4u0r)

> For the processed dataset, you should directly put the dataset in './data/' and unpack it. For the original dataset, you should put it in './data/' and run 'python preprocess.py --dataset_name NL'.

## Results
<p align="center"> 
    <img src="https://github.com/yjdy/Multitask-Optimization-Recommendation-Library/blob/main/misc/result.png" width="800">
</p>

## Citation

If our work has been of assistance to your work, feel free to give us a star ⭐ or cite us using :  

```
@article{yuan2024parameterupdatebalancingalgorithm,
      title={A Parameter Update Balancing Algorithm for Multi-task Ranking Models in Recommendation Systems}, 
      author={Jun Yuan and Guohao Cai and Zhenhua Dong},
      journal={arXiv preprint arXiv:2410.05806},
      year={2024},
      eprint={2410.05806},
      archivePrefix={arXiv}
}
```

> Thanks to the excellent code repository [Multitask-Recommendation-Library](https://github.com/easezyc/Multitask-Recommendation-Library), [NashMTL](https://github.com/AvivNavon/nash-mtl) and [FAMO](https://github.com/Cranial-XIX/FAMO)! 
> MTOReclib is released under the MIT License, some codes are modified from FAMO and Multitask-Recommendation-Library, which are released under the Apache License 2.0 and MIT License, respectively.
