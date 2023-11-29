# Knowledge Distillation(teacher-student model) for solving the class imbalanced problem in Image classification


![Screenshot 2023-11-29 231934](https://github.com/shivam3110/kd_thesis/assets/56818878/78cbb0de-b97c-47c2-ba31-72b989ef8a38)

This repository is dedicated to the implementation of Knowledge Distillation (KD) techniques, specifically utilizing the teacher-student model, to address class imbalance challenges in image classification. The primary focus is on applying these methodologies to medical image classification, with an emphasis on the Brain Tumor Segmentation dataset (BraTS2020)


Dataset: **Brain Tumor Segmentation(BraTS2020)**  
          https://www.kaggle.com/datasets/awsaf49/brats2020-training-data?resource=download

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
This repository explores the potential of Knowledge Distillation (KD), specifically the teacher-student model, to improve the detection of minority classes, focusing on the Brain Tumor Segmentation dataset (BraTS2020) with multi-modal MRI scans

### Overview:
Class-imbalanced datasets pose a formidable challenge in machine learning, especially in tasks like image classification. The inherent skew towards majority classes often results in models favoring these classes, potentially neglecting crucial information from the minority class. This bias can lead to models struggling to accurately predict underrepresented classes, which is crucial in various real-world applications.

![image](https://github.com/shivam3110/kd_thesis/assets/56818878/3474363d-b5e1-4e0d-ae2b-65c2aa909bd7)

**Knowledge Distillation (KD) for the minority class**
“Knowledge Distillation (KD) is a technique in machine learning that involves training a simpler model, known as the student, to mimic the behavior of a more complex model, known as the teacher.”
Knowledge Distillation is a machine learning technique involving the training of a simpler model (student) to mimic the behavior of a more complex model (teacher).


### Goals:
1. Can Knowledge Distillation(teacher-student model) address the problem of class imbalance for Image Classification?
2. How does the selection of teacher and student models affect the performance based on the number of model parameters?



### Installation

Provide step-by-step instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/shivam3110/kd_thesis.git

# Change into the project directory
cd kd_thesis

# Install dependencies
requirement.txt
```

### Results

#### 1. KFOLD baseline performance(F1 score)
![image](https://github.com/shivam3110/kd_thesis/assets/56818878/a3962cb9-f830-4fcb-9f7b-30895a24f157)
#### 2. KFOLD knowledge distillation performance(F1 score)
![image](https://github.com/shivam3110/kd_thesis/assets/56818878/c19924fc-3a11-443a-95df-8e939aa4092e)


   



### Contributing

Feel free to contribute by opening [issues](https://github.com/shivam3110/kd_thesis/issues), providing feedback, or submitting [pull requests](https://github.com/shivam3110/kd_thesis/pulls).

### License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE.md](https://github.com/shivam3110/kd_thesis/blob/main/LICENSE.md) file for details.

### Acknowledgments

Express gratitude and acknowledge any contributions, tools, or libraries used in the project.


