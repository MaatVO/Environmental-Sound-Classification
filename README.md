# Environmental Sound Recognition using CNNs

## Project Overview
This project explores the effectiveness of different input features in Convolutional Neural Network (CNN) architectures for Environmental Sound Recognition. We compare three different CNN models:

1. MelNet: Uses log-mel feature input
2. RawNet: Processes raw waveform inputs
3. MFCC Model: Based on Mel-frequency cepstral coefficients

The models are benchmarked on the ESC-50 dataset, a widely used dataset for environmental sound classification.

## Key Features
- Implementation of three distinct CNN architectures
- Comparison of log-mel features, raw waveform input, and MFCCs
- Use of a five-layer stacked CNN network with decreasing filter sizes
- End-to-end stacked CNN model for raw waveform processing
- Evaluation using the ESC-50 dataset

## Results
Our experiments yielded interesting results, with the MFCC model outperforming the others:

| Model  | Test Accuracy |
|--------|---------------|
| MFCC   | 72% ± 3       |
| MelNet | 69% ± 3       |
| RawNet | 62% ± 3       |

## Methodology
- Data preprocessing for each model type
- Training with dynamic learning rate and batch size optimization
- Evaluation using majority voting rule
- Use of "Sparse Categorical Crossentropy" as the loss function

## Conclusions
The study demonstrates the significance of selecting appropriate input features for enhancing model performance in environmental sound recognition tasks. The MFCC model showed the best performance, suggesting that MFCC features provide a more robust representation of sound characteristics for this task.

## Future Work
- Explore hybrid models combining strengths of different input features
- Investigate impact of data augmentation techniques
- Apply transfer learning to improve model performance

## Technologies Used
- Python
- TensorFlow
- Librosa
- NumPy
- Matplotlib

## Contributors
- Mattia Varagnolo

## License
[Include license information]

## Acknowledgements
This project builds upon the work of Shaobo Li et al. and other researchers in the field of environmental sound classification.
