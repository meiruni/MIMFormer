
# MIMFormer:Multiscale Inception Mixer Transformer for  Hyperspectral and Multispectral Image Fusion




## Network Architecture

- The overall architecture diagram of our proposed MIMFormer network.
  ![MIMFormer](./MIMFormer.png)


- The architecture diagram of the Inception Spatial Spectral Mixer.
  ![ISSM](./ISSM.png)

## 1. Create Envirement:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

[pip install timm
pip install thop
pip install scikit-image
pip install scipy
pip install einops
pip install opencv-python
pip install tensorboard]()

## 2. Data Preparation:

- Download the CAVE dataset from <a href="https://www1.cs.columbia.edu/CAVE/databases/multispectral">here</a>.
- Download the PU dataset from <a href="[Hyperspectral Remote Sensing Scenes - Grupo de Inteligencia Computacional (GIC) (ehu.eus)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)">here</a>.
- Download the WDCM dataset from <a href="[MSST-Net/README.md at main · jx-mzc/MSST-Net · GitHub](https://github.com/jx-mzc/MSST-Net/blob/main/README.md)">here</a>.
-  the real remote sensing dataset ZY1E, which was used in this study, has been completely uploaded and can be accessed through the following link: [https: //pan.baidu.com/s/1qoY9tQF0mwlgxy6hTk5ebA?pwd=7g68](). The extraction code to access the dataset is 7g68, and it is publicly available

## 3.How to do

Place the data files into the "Datasets" folder. Simply run the "main.py" file to execute.

## Attention

If you have any questions, please feel free to contact me.    Email:2220902206@cnu.edu.cn
