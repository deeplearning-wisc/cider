# How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection? 

This codebase provides a Pytorch implementation for the paper CIDER: [How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?](https://openreview.net/forum?id=aEFaE0W5pAd) at ICLR 2023.

### Abstract

Out-of-distribution (OOD) detection is a critical task for reliable machine learning. Recent advances in representation learning give rise to distance-based OOD detection, where testing samples are detected as OOD if they are relatively far away from the centroids or prototypes of in-distribution (ID) classes. However, prior methods directly take off-the-shelf contrastive losses that suffice for classifying ID samples, but are not optimally designed when test inputs contain OOD samples. In this work, we propose CIDER, a novel representation learning framework that exploits hyperspherical embeddings for OOD detection. CIDER jointly optimizes two losses to promote strong ID-OOD separability: a dispersion loss that promotes large angular distances among different class prototypes, and a compactness loss that encourages samples to be close to their class prototypes. We analyze and establish the unexplored relationship between OOD detection performance and the embedding properties in the hyperspherical space, and demonstrate the importance of dispersion and compactness. CIDER establishes superior performance, outperforming the latest rival by 19.36% in FPR95. 

### Illustration

![fig1](readme_figs/fig1.png)



## Quick Start

Remarks: This is the initial version of our codebase, and while the scripts are functional, there is much room for improvement in terms of streamlining the pipelines for better efficiency. Additionally, there are unused parts that we plan to remove in an upcoming cleanup soon. Stay tuned for more updates.

### Data Preparation

The default root directory for ID and OOD datasets is `datasets/`. We consider the following (in-distribution) datasets: CIFAR-10, CIFAR-100, and ImageNet-100. 

**Large-scale OOD datasets** For large-scale ID (e.g. ImageNet-100), we use the curated 4 OOD datasets from [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf), and de-duplicated concepts overlapped with ImageNet-1k. The datasets are created by  [Huang et al., 2021](https://github.com/deeplearning-wisc/large_scale_ood) .

The subsampled iNaturalist, SUN, and Places can be download via the following links:

```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz

```
The directory structure looks like:
```python
datasets/
---ImageNet100/
---ImageNet_OOD_dataset/
------dtd/
------iNaturalist/
------Places/
------SUN/
```

**Small-scale OOD datasets** For small-scale ID (e.g. CIFAR-10), we use SVHN, Textures (dtd), Places365, LSUN-C (LSUN), LSUN-R (LSUN_resize), and iSUN. 

The directory structure looks like:

```python
datasets/
---CIFAR10/
---CIFAR100/
---small_OOD_dataset/
------dtd/
------iSUN/
------LSUN/
------LSUN_resize/
------places365/
------SVHN/
```



## Training and Evaluation 

### Model Checkpoints

**Evaluate pre-trained checkpoints** 

Our checkpoints can be downloaded here for [CIFAR-100](https://drive.google.com/drive/folders/1SjW2kvhDQ6qcsIo5TR7eLMrcL3r6Y3QN?usp=share_link) and [CIFAR-10](https://drive.google.com/drive/folders/1rkXQYHcaITZCj55OLNXqy_b-yjktONrn?usp=share_link). Create a directory named `checkpoints/{ID}` in the root directory of the project and put the downloaded checkpoints here:

```
checkpoints/
---CIFAR-10/	 	
------ckpt_c10/
------checkpoint_500.pth.tar
---CIFAR-100/	 	
------ckpt_c100/
------checkpoint_500.pth.tar
```

The following scripts can be used to evaluate the OOD detection performance:

```
sh scripts/eval_ckpt_cifar10.sh ckpt_c10 #for CIFAR-10
sh scripts/eval_ckpt_cifar100.sh ckpt_c100 # for CIFAR-100
```



**Evaluate custom checkpoints** 

The default directory to save checkpoints is `checkpoints`. Otherwise, create a softlink to the directory where the actual checkpoints are saved and name it as `checkpoints`). For example, checkpoints for CIFAR-100 (ID) are accessible in the `checkpoints` directory with the following structure: 

```python
checkpoints/
---CIFAR-100/
------name_of_ckpt/
---------checkpoint_5.pth.tar
---------checkpoint_10.pth.tar
```



**Train from scratch** 

We provide sample scripts to train from scratch. Feel free to modify the hyperparameters and training configurations.

```
sh scripts/train_cider_cifar10.sh
sh scripts/train_cider_cifar100.sh
sh scripts/train_cider_imgnet100.sh
```




### Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{ming2023cider,
 title={How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?},
 author={Yifei Ming and Yiyou Sun and Ousmane Dia and Yixuan Li},
 booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=aEFaE0W5pAd}
}
```


