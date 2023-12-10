# Environmental Sound Classification

## Overview

This project focuses on the classification of environmental sounds using the ESC-50 dataset. The classification is performed by employing an ensemble of two distinct Convolutional Neural Networks (CNNs). One CNN utilizes raw waveform data, while the other utilizes log-spectrogram data. These models are integrated into an ensemble using Dempster-Shafer (DS) theory.

## Table of Contents

- [Introduction](#environmental-sound-classification)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Ensemble](#ensemble)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The project is based on the ESC-50 dataset, which consists of 2,000 environmental sound recordings. Each recording belongs to one of 50 classes, making it a valuable resource for training and evaluating sound classification models. You can find more information about the dataset [here](https://github.com/karolpiczak/ESC-50).
This article presents an implementation trial of the research paper titled "An Ensemble Stacked Convolutional Neural Network Model for Environmental Event Sound Recognition" by Shaobo Li, Yong Yao, Jie Hu, Guokai Liu, Xuemei Yao, and Jianjun Hu. The trial was conducted by researchers from the School of Mechanical Engineering at Guizhou University and the Department of Computer Science and Engineering at the University of South Carolina.
The paper can be found [here](https://www.mdpi.com/2076-3417/8/7/1152?type=check_update&version=1)
## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/environmental-sound-classification.git

2. Install the required dependencies (WIP)

   ```bash
   pip install -r requirements.txt

## Usage
Run the ipynb file. (WIP)

## Models
The Convolutional Neural Network (CNN) architecture utilized in this project comprises several layers, each designed to capture distinct features of environmental sounds. The model follows a unique configuration, including elements inspired by the VGG network and customized choices to enhance learning capabilities.

_Layer 1 (L1)_

- **Filters:** 24
- **Receptive Field:** (6,6)
- **Stride:** (1,1)
- **Activation Function:** ReLU
- **Purpose:** Learn smaller and more local "time-frequency" characteristics of sound segments.

_Layer 2 (L2)_

- **Filters:** 24
- **Receptive Field:** (6,6)
- **Stride:** (1,1)
- **Activation Function:** ReLU
- **Purpose:** Similar to VGG network, enables the network to learn high-level features.

_Layer 3 (L3)_

- **Filters:** 48
- **Receptive Field:** (5,5)
- **Stride:** (2,2)
- **Activation Function:** ReLU

_Layer 4 (L4)_

- **Filters:** 48
- **Receptive Field:** (5,5)
- **Stride:** (2,2)
- **Activation Function:** ReLU

## Layer 5 (L5)

- **Filters:** 64
- **Receptive Field:** (4,4)
- **Stride:** (2,2)
- **Activation Function:** ReLU

_Layer 6 (L6)_

- **Type:** Fully Connected Layer
- **Hidden Units:** 200
- **Activation Function:** ReLU
- **Note:** Applied a 0.5 dropout probability for the fully connected layer in RawNet to prevent overfitting.

_Layer 7 (L7)_

- **Output Units:** 10 or 50
- **Activation Function:** Softmax
- **Note:** The output represents the number of classes, and the choice between 10 or 50 units depends on the specific task.

In contrast to the standard CNN configuration, no max-pooling layer is adopted after the convolutional layers of the recognition modules in either RawNet or MelNet. Instead, a Batch Normalization layer is used to accelerate and improve the learning process of deep neural networks. Additionally, the Batch Normalization layer is followed by a fully connected layer. A learning decay scheme is applied during the training process to optimize model training.

This architecture is designed to effectively capture diverse features within environmental sound data and maintain detailed information through the avoidance of max-pooling in favor of Batch Normalization. These choices aim to enhance the model's performance in sound classification tasks.

### RawNet
  
  
### MelNet
  

### DS enseble
