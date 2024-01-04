# Lung Cancer Diagnosis with Medical Imaging

## Introduction
Welcome to my Lung Cancer Diagnosis project. This research aims to enhance the accuracy and speed of lung cancer diagnosis using advanced machine learning and deep learning techniques. The project utilizes the Lung and Colon Cancer Histopathological Image Dataset (LC25000), containing 25,000 color images associated with five histologic entities. For computational reasons, the colon tissues were excluded from this study.

<table>
  <tr>
    <td><img src="https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f1.png" alt="Figure 1" /></td>
    <td><img src="https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f2.png" alt="Figure 2" /></td>
  </tr>
</table>

### Problem Statement and Motivation
The central research question is whether machine learning and computer vision can improve the accuracy and efficiency of lung cancer diagnosis. This project is motivated by the potential to revolutionize cancer diagnosis through the application of cutting-edge technologies. 

## Methods
The project explores the trade-offs between traditional machine learning (ML) and deep learning (DL) models. Traditional ML models perform well with tabular data but are less effective with complex, high-dimensional medical images. In contrast, deep learning models, particularly Convolutional Neural Networks (CNNs), are highly effective with images, learning intricate patterns and spatial hierarchies.

### Machine Learning

By projecting our image dataset into a 2-dimensional space using PCA, we observe distinct clusters where each point corresponds to an image, and the color coding represents different classes. The evident grouping implies that images within the same class have similar key features, and these features vary significantly between classes in the reduced dimensionality space. This clear separation between classes suggests that even traditional machine learning algorithms could be sufficient to achieve high accuracy. 

![Figure 3](https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f3.png)

### Deep Learning

Despite the distinct clusters observed in the PCA projection, the project extends into deep learning architectures to enhance performance and understanding. The focus is on VGG16 and ResNet18, two prominent convolutional neural network models. VGG16 is examined for its depth and architectural simplicity, followed by an investigation of ResNet18, which is notable for its residual blocks that facilitate the training of significantly deeper networks.

<table>
  <tr>
    <td><img src="https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f4.png" alt="Figure 1" /></td>
    <td><img src="https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f5.png" alt="Figure 2" /></td>
  </tr>
</table>

## Results

### VGG16 Training Dynamics
We monitor the training and validation process of VGG16 across epochs to ensure an increase in testing accuracy over epochs. 5-Fold cross-validation is employed demonstrating the model's ability to learn and generalize from the data effectively.

![Training and Testing Accuracy Over Epochs](https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f7.png)

### Comparative Analysis of Model Accuracy
The evaluation of various classifiers shows a range of accuracies, with deep learning models outperforming traditional machine learning counterparts. Notably, VGG16 achieved a significant accuracy of 0.94 ± 0.01, while ResNet18 topped the comparison with an impressive accuracy of 0.98 ± 0.00. This indicates the effectiveness of deep learning architectures in handling complex image data over more traditional models.

![Figure 6](https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Figures/f6.png)

Other performance metrics may be used, however the balanced nature of our dataset prompts simplicity.










### Project Files Overview:
- `ml.py`, `resnet.py`, `vgg.py`: Implementation of various ML algorithms and DL models.
- Job scripts and outputs: `my_job.sub`, `resnet.sub`, `output.txt`, `resnet.txt`.
- `reader.py`: Script for loading the dataset/images.
- `test.py`: Inference script for the DL models.
- `Written Report.pdf`: A comprehensive overview of the problem, motivation, methods, results, findings, and conclusions.
- [LC25000 dataset paper](https://arxiv.org/pdf/1912.12142.pdf)
- [Full Written Report](https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Written%20Report.pdf)

## Acknowledgements
I would like to thank everyone involved in the Lung and Colon Cancer Histopathological Image Dataset (LC25000) and the AI researchers contributing to cancer pathology. Their efforts have been instrumental in advancing research in this critical domain.

---
**Note:** For more detailed information and a comprehensive understanding of the project, please refer to the attached `Written Report.pdf` or click here [Full Written Report](https://github.com/napronald/Lung-Cancer-Diagnosis-with-Medical-Imaging/blob/main/Written%20Report.pdf).
