# LensDroid

This repository contains the source code for the paper [Detecting Android Malware by Visualizing App Behaviors From Multiple Complementary Views](https://ieeexplore.ieee.org/document/10908919?source=authoralert). The project is open for academic communication and further development.

## Paper Information

- **Title:** Detecting Android Malware by Visualizing App Behaviors From Multiple Complementary Views 
- **Authors:** Zhaoyi Meng; Jiale Zhang; Jiaqi Guo; Wansen Wang; Wenchao Huang; Jie Cui et al.  
- **Published in:** IEEE  
- **Paper Link:** [https://ieeexplore.ieee.org/document/10908919?source=authoralert](https://ieeexplore.ieee.org/document/10908919?source=authoralert)

## Project Overview

Deep learning has emerged as a promising technology for achieving Android malware detection. To further unleash its detection potentials, software visualization can be integrated for analyzing the details of app behaviors clearly. However, facing increasingly sophisticated malware, existing visualization-based methods, analyzing from one or randomly-selected few views, can only detect limited attack types. We propose and implement LensDroid, a novel technique that detects Android malware by visualizing app behaviors from multiple complementary views. Our goal is to harness the power of combining deep learning and software visualization to automatically capture and aggregate high-level features that are not inherently linked, thereby revealing hidden maliciousness of Android app behaviors. To thoroughly comprehend the details of apps, we visualize app behaviors from three related but distinct views of behavioral sensitivities, operational contexts and supported environments. We then extract high-order semantics based on the views accordingly. To exploit semantic complementarity of the views, we design a deep neural network based model for fusing the visualized features from local to global based on their contributions to downstream tasks. A comprehensive comparison with six baseline techniques is performed on datasets of more than 51K apps in three real-world typical scenarios, including overall threats, app evolution and zero-day malware. The experimental results show that the overall effectiveness of LensDroid is better than the baseline techniques. We also validate the complementarity of the views and demonstrate that the multi-view fusion in LensDroid enhances Android malware detection.


## Directory Structure
- `script_preprocess/`: Extract three view features corresponding to the data set
- `network/`: Dataset training test 
- `TrainData/`: Dataset(2018 to 2022) 
- `README.md`: Project introduction  
- `LICENSE`: License file

 
## Requirements

- Operating System: Linux
- Python version: 3.11

## Source Code

1. Extract features：

    1.1Extract Artifacts features
   ```bash
   python ./script_preprocess/apk2img_Decorator.py
   python ./script_preprocess/apk2img.py
   ```

     1.2Extract API calls features
   ```bash
   sh ./script_preprocess/generateCG_begin.sh
   sh ./script_preprocess/CGtoVector_begin.sh
   ```

     1.3Extract Opcode features
   ```bash
   python ./script_preprocess/opcode1_apk2smali.py
   sh ./script_preprocess/opcode2_begin.sh
   python ./script_preprocess/opcode3_opcodeSeq2id.py
   python ./script_preprocess/opcode4_opcodeID2oneHot.py
   ```
2. Run the [train](https://github.com/2315433606/LensDroid/blob/main/network/train.py) (see more configurations in the code)::
   ```bash
   python ./network/train.py
   ```

## Dataset and Experiments

To reproduce the experiments in the paper, please contact the authors for related datasets and scripts.


## Citation

If you find this work useful for your research, please consider citing our [paper](https://ieeexplore.ieee.org/document/10908919?source=authoralert):

```bibtex
@ARTICLE{10908919,
  author={Meng, Zhaoyi and Zhang, Jiale and Guo, Jiaqi and Wang, Wansen and Huang, Wenchao and Cui, Jie and Zhong, Hong and Xiong, Yan},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Detecting Android Malware by Visualizing App Behaviors From Multiple Complementary Views}, 
  year={2025},
  volume={20},
  number={},
  pages={2915-2929},
  keywords={Malware;Visualization;Semantics;Feature extraction;Sensitivity;Codes;Vectors;XML;Image edge detection;Training;Mobile security;Android malware detection;behavioral visualization;multi-view learning},
  doi={10.1109/TIFS.2025.3547301}}

```







