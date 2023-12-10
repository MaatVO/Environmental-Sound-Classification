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

_Layer 5 (L5)_

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
The feature learning module in RawNet, as illustrated in Figure 1a, is meticulously designed to extract meaningful features from the raw waveform input. Comprising two convolutional layers preceding a pooling layer, this module aims to capture essential two-dimensional information from the input data. Each convolutional layer, equipped with 40 filters, functions akin to a bandpass filter bank, emulating log-scaled mel-spectrogram components that cover the audible frequency range (0–22,050 Hz) of the sound segment, approximately 1 second in duration.

The design of each convolutional layer incorporates small receptive fields of (1,8) and a time series stride of (1,1). This choice aligns with insights from EnvNet [5], highlighting the efficacy of CNN models in hierarchically extracting local features of diverse time scales using multiconvolutional layers with small receptive fields. The input shape and parameters of the first two convolutional layers are concisely summarized in Table 1.

**Parameters of the Feature Learning Module in RawNet (Table 1)**

| Layer  | Input Shape          | Filter | Kernel Size | Stride | Output Shape           |
|--------|----------------------|--------|-------------|--------|------------------------|
| Conv1  | [batch,1,20480,1]    | 40     | (1,8)        | (1,1)  | [batch,1,20480,40]     |
| Conv2  | [batch,1,20480,40]   | 40     | (1,8)        | (1,1)  | [batch,1,20480,40]     |
| Pool   | [batch,1,20480,40]   | 40     | (1,128)      | (1,128)| [batch,1,160,40]       |

The output of these two convolutional layers forms the "time-frequency" feature representation. Subsequently, a non-overlapping max-pooling operation is applied to the output with a pooling size of 128. The resulting matrix, with dimensions 1 × 160 × 40 (frequency × time × channel), undergoes a reshaping process to attain dimensions of 40 × 160 × 1. This reshaped matrix is then seamlessly fed into the convolutional layer "conv3" to facilitate the final classification.

The meticulously crafted architecture of this module ensures the capture of pertinent time-frequency features from the raw waveform input, laying a robust foundation for subsequent classification tasks. The parameter choices, such as small filter sizes and non-overlapping max pooling, are grounded in experimental insights, optimizing the extraction of hierarchical local features. 
  
### MelNet
For log-mel feature extraction, leveraging the capabilities of the librosa Python library, we extract log-scaled mel-spectrogram features employing 60 bands, effectively covering the frequency range (0–22,050 Hz) of the sound segments. Simultaneously, the sound segments undergo division into 41 frames with a 50% overlap, resulting in each frame spanning approximately 23 ms. These procedural steps yield a static log-scaled mel-spectrogram feature representation for each segment, manifested as a 60 × 40 × 1 matrix, corresponding to frequency × time × channel.

Moreover, we calculate the first temporal derivative of the log-mel feature on each frame, generating the delta log-scaled mel-spectrogram feature. This derivative serves as the second channel of input. Consequently, the dimensionality of the extracted log-scaled mel-spectrogram feature maps is consolidated into a 60 × 41 × 2 matrix.

This feature extraction approach not only captures the static characteristics of the mel-spectrogram but also incorporates dynamic temporal information through the calculation of temporal derivatives. The resulting feature representation forms a comprehensive 3D matrix, ready for integration into the subsequent stages of our sound classification model.


### Using Dempster-Shafer (DS) Evidence Theory for Sound Classification

The fundamental concept of DS evidence theory involves establishing a frame of discernment, denoted as Θ, consisting of a finite set of elements {A1, A2, ..., An}, where n represents the number of elements. Each element is considered incompatible and independent. The power set of Θ, denoted as 2Θ, plays a crucial role in DS theory.

In DS evidence theory, the basic probability assignment function, denoted as m, maps the power set 2Θ to the interval [0, 1]. The following conditions must be satisfied:
1. m(∅) = 0: The probability of an impossible event is 0.
2. ∑A⊆Θ m(A) = 1: The total probability of events within the frame of discernment is 1.

The function m(A), referred to as the basic probability assignment (BPA) of the element A, reflects the degree of trust in the element itself.

In the context of sound classification, each class of sounds in the dataset corresponds to an element in {A1, A2, ..., An} under the frame of discernment, with n being the number of classes (e.g., n = 10 for ESC-10 and Urbansound8k; n = 50 for ESC-50). All elements are exclusive and independent.

The activation functions' (softmax) output values from the MelNet and RawNet models serve as the basic probability assignments m1 and m2, respectively, under the same frame of discernment, Θ. These assignments satisfy the conditions:
1. 0 ≤ m(A) ≤ 1, ∀A ⊂ Θ
2. ∑A⊆Θ m(A) = 1

An orthogonal operation is then employed to effectively synthesize the basic probability assignments generated by the two models. For any A ∈ Θ, the fusion formula is given by:
\[ (m1 ⊕ m2)(A) = \frac{1}{K} \sum_{Ai ∩ Aj = ∅} m1(Ai) \cdot m2(Aj) \]

Here, K is a coefficient measuring the degree of conflict between various pieces of evidence. The output value (m1 ⊕ m2)(A) is a basic probability assignment and satisfies \(\sum_{A⊂Θ} (m1 ⊕ m2)(A) = 1\), serving as the comprehensive probability assignment of m1 and m2. This output is utilized as the final prediction result of the fusion, providing a unified and normalized probability distribution across the sound classes.

