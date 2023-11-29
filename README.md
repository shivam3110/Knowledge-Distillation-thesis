# Knowledge Distillation(teacher-student model) for solving the class imbalanced problem in Image classification

This repository contains code and resources related to my knowledge distillation for class-imbalanced datasets. The project focuses on knowledge distillation techniques applied to Medical image classification.


Dataset: **Brain Tumor Segmentation(BraTS2020)**  https://www.kaggle.com/datasets/awsaf49/brats2020-training-data?resource=download

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
Imbalanced datasets pose a significant challenge in machine learning, particularly in tasks like image classification. They often lead to a skewed learning process where the model tends to favor the majority class, potentially overlooking crucial information from the minority class. This discrepancy can lead to models that struggle to accurately predict the underrepresented class, which is crucial in many real-world applications.

**Knowledge Distillation (KD) for the minority class**
“Knowledge Distillation (KD) is a technique in machine learning that involves training a simpler model, known as the student, to mimic the behavior of a more complex model, known as the teacher.”
%Provide a brief overview of the project, its goals, and the problem it aims to solve. Include any relevant background information to help users understand the context of your thesis.

### Goals:
1. Can Knowledge Distillation(teacher-student model) address the problem of class imbalance for Image Classification?
2. How does the selection of teacher and student models affect the performance based on the number of model-parameters?




## Getting Started

Explain how to get a copy of the project up and running on a local machine.

### Prerequisites

List any software, libraries, or dependencies that users need to have installed before they can run your project.

### Installation

Provide step-by-step instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/shivam3110/kd_thesis.git

# Change into the project directory
cd kd_thesis

# Install dependencies
# Using file requirement.txt
