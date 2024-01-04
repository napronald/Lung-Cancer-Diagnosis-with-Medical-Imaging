# Lung Cancer Diagnosis with Medical Imaging

## Introduction
Welcome to my Lung Cancer Diagnosis project. This research aims to enhance the accuracy and speed of lung cancer diagnosis using advanced machine learning and deep learning techniques. The project utilizes the Lung and Colon Cancer Histopathological Image Dataset (LC25000), containing 25,000 color images associated with five histologic entities.

## Problem Statement and Motivation
The central research question is whether machine learning and computer vision can improve the accuracy and efficiency of lung cancer diagnosis. This project is motivated by the potential to revolutionize cancer diagnosis through the application of cutting-edge technologies.

## Dataset
The dataset consists of 25,000 images, three classes related to lung cancer and two to colon cancer. Due to computational and time constraints, this study focused only on lung cancer tissues, using 15,000 images for the experiments. More details about the dataset can be found in the [LC25000 dataset paper](https://arxiv.org/pdf/1912.12142.pdf).

## Methods
The project explores the trade-offs between traditional machine learning (ML) and deep learning (DL) models. Traditional ML models perform well with tabular data but are less effective with complex, high-dimensional medical images. In contrast, deep learning models, particularly Convolutional Neural Networks (CNNs), are highly effective with images, learning intricate patterns and spatial hierarchies.

### Project Files Overview:
- `ml.py`: Code for using various ML algorithms.
- `my_job.sub`: Job script for the VGG model.
- `output.txt`: Output for the VGG model.
- `reader.py`: Loads the necessary CSV file for the dataset/images.
- `resnet.py`: Code for the ResNet-18 model.
- `resnet.sub`: Job script for the ResNet model.
- `resnet.txt`: Output of the ResNet model.
- `test.py`: Inference code for the VGG model.
- `vgg.py`: Code for the VGG-16 model.
- `Written Report.pdf`: A comprehensive overview of the problem, motivation, methods, results, findings, and conclusions.

## Results
The study reveals that traditional machine learning models achieved commendable accuracy, and deep learning models, particularly CNNs, were well-suited for medical image analysis due to their ability to capture intricate patterns and spatial relationships within images.

## Limitations
This study acknowledges its inherent limitations, including the size of the dataset, computational constraints, and the potential discrepancy in comparison due to the application of Principal Component Analysis (PCA) only to machine learning methods.

## Conclusion
The deployment of deep convolutional neural networks marked a significant breakthrough, showcasing their pivotal role in the development of robust clinical decision support systems. These advanced networks offer exciting prospects for revolutionizing the accuracy and speed of lung cancer diagnosis, representing a significant advancement in medical imaging technology.

## Acknowledgements
I would like to thank everyone involved in the Lung and Colon Cancer Histopathological Image Dataset (LC25000) and the AI researchers contributing to cancer pathology. Their efforts have been instrumental in advancing research in this critical domain.

---
**Note:** For more detailed information and a comprehensive understanding of the project, please refer to the attached `Written Report.pdf`.
