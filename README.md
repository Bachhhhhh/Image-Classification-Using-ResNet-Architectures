# Image Classification Using ResNet Architectures

---

## Table of Contents

* [Project Overview](#project-overview)
  * [Overview of ResNet](#overview-of-resnet)
  * [Dataset Description](#dataset-description)
* [Requirements](#requirements)
* [Workflow](#workflow)
  * [1. Reproducibility Setup](#1-reproducibility-setup)
  * [2. Train-Test Split](#2-train-test-split)
  * [3. Custom Dataset Definition](#3-custom-dataset-definition)
  * [4. Image Transformations](#4-image-transformations)
  * [5. DataLoader Definition](#5-dataloader-definition)
  * [6. Residual Block Definition](#6-residual-block-definition)
  * [7. ResNet Model Definition](#7-resnet50-model-definition)
  * [8. Callbacks Definition](#8-callbacks-definition)
  * [9. Model Training & Evaluation](#9-model-training--evaluation)
* [Conclusion](#conclusion)
---

## Project Overview

### Overview of ResNet

Residual Network (ResNet) is a famous deep learning architecture introduced to address the vanishing gradient problem in very deep neural networks. 
The core idea of ResNet is the use of residual (skip) connections, which allow gradients to flow directly through the network by bypassing one or more layers. 
This enables the successful training of very deep models such as ResNet50, ResNet101, and beyond.

The picture below provides a general overview of ResNet architecture

<img width="1103" height="445" alt="ResNet Architectures" src="https://github.com/user-attachments/assets/39902503-01c3-4487-a1f0-f53b48f3bba6" />

---

### Dataset Description

#### CIFAR-10 Dataset

This project uses the **CIFAR-10-images-master** dataset, which consists of 60,000 color images (32×32 pixels) across 10 classes, including airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, and truck.

#### Download and Unzip Dataset

The dataset can be downloaded and extracted using the following commands:

```bash
!gdown --quiet 1UL2FYpZmqvyFchFReFVJEi3V47F8ksv_
!unzip -q /content/CIFAR-10-images-master.zip
```

The dataset is organized into two main folders:

* `train/`: Training images
* `test/`: Testing images

Example images from the dataset:

<img width="543" height="271" alt="image" src="https://github.com/user-attachments/assets/3e6ad296-749f-4d9e-aa0f-1972bad7a37d" />

---

## Requirements

The following Python libraries are required to run this project:

* `glob`
* `pandas`
* `numpy`
* `os`
* `lightning`
* `torch`
* `torchmetrics`
* `torchvision`
* `matplotlib`
* `scikit-learn`

---

## Workflow

### 1. Reproducibility Setup

To ensure reproducible results across multiple runs, a global random seed is set for all relevant libraries.

```python
def set_seed(seed: int = 0) -> None:
    """Set random seed for reproducibility across numpy, random, torch, and CUDA."""
```

This function synchronizes randomness across NumPy, Python's `random`, PyTorch CPU, and CUDA backends.

---

### 2. Train–Test Split

Two main dataset paths are defined:

```python
test_path = "/content/CIFAR-10-images-master/test/"
train_path = "/content/CIFAR-10-images-master/train/"
```

A helper function is used to collect image paths and labels:

```python
def get_path(path):
    """Generate a list of dictionaries with image paths, labels, and class names."""
```

The dataset is then split into training and validation sets:

```python
train_paths = get_path(train_path)
test_paths = get_path(test_path)
train_paths, val_paths = train_test_split(
    train_paths, test_size=0.2, random_state=0
)
```

---

### 3. Custom Dataset Definition

A custom PyTorch `Dataset` class is defined for flexible image loading.

```python
class DataSet(torch.utils.data.Dataset):
    """Custom Dataset for loading images and their corresponding labels."""
```

#### Key Methods

* **`__init__`**: Initialize image paths, labels, and optional transformations.
* **`__len__`**: Return the total number of samples.
* **`__getitem__`**: Load and return an image–label pair as a dictionary.

---

### 4. Image Transformations

Separate transformation pipelines are defined for training and validation/testing datasets.

#### Transformation Pipeline

1. Convert image to PIL format
2. Resize image to **224 × 224**
3. Convert image to tensor
4. Normalize using **ImageNet mean and standard deviation**

This ensures compatibility with ImageNet-pretrained ResNet architectures.

---

### 5. DataLoader Definition

DataLoaders handle batching, shuffling, and parallel data loading efficiently.

```python
batch_size = 32
num_workers = 2
```

```python
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
)
```

---

### 6. Residual Block Definition

A **Residual Block** is the fundamental building unit of ResNet. The residual connection can be expressed as:

y = F(x) + x

where (F(x)) represents the output of stacked convolutional layers, and (x) is the identity mapping.

<img width="566" height="393" alt="Residual Block Visualization" src="https://github.com/user-attachments/assets/618c6ac0-f503-4c0f-a476-1dd80abcd203" />

### Architecture

* **1×1 Convolution**: Reduce channel dimension to `c_out`
* **3×3 Convolution**: Feature extraction at constant channel size
* **1×1 Convolution**: Expand channels to `c_out × 4`

#### Forward Pass Logic

* Save input x as identity
* Pass input through convolutional block
* Add identity (skip connection)
* Apply ReLU activation
  
---

### 7. ResNet50 Model Definition

```python
class ResNet50(L.LightningModule):
```

#### Initial Channel Size

```python
self.in_channels = 64
```

The value **64** is chosen to match the original ResNet design and to provide sufficient representational capacity at early stages.

#### Initial Layers

* **7×7 Convolution, 64 channels, stride 2**: Capture low-level features
* **Batch Normalization**: Stabilize training
* **ReLU**: Introduce non-linearity
* **3×3 MaxPooling, stride 2**: Reduce spatial dimensions

#### ResNet Stages

* Layer 1: 3 blocks
* Layer 2: 4 blocks
* Layer 3: 6 blocks
* Layer 4: 3 blocks

#### Classification Head

* **Adaptive Average Pooling**: Reduce spatial dimensions to 1×1
* **Fully Connected Layer**: Map features to class scores

#### Layer Construction Helper

```python
def _make_layer(self, out_channels, blocks, stride=1):
```

This method handles downsampling and stacks multiple residual blocks into a sequential layer.

#### Forward Method

```python
def forward(self, x):
```

Execution flow:

1. Initial convolution and pooling
2. Pass through ResNet layers
3. Pool, flatten, and classify

#### Training Logic

The model defines:

* `training_step`
* `validation_step`
* `test_step`
* `predict_step`

#### Optimizer

```python
def configure_optimizers(self):
```

Uses **Adam optimizer** with learning rate **1e-3**.

---

### 8. Callbacks Definition

#### Model Checkpoint

```python
model_checkpoint = ModelCheckpoint(
    dirpath='checkpoint/',
    monitor="val_acc",
    verbose=True,
    mode="max",
    save_top_k=1
)
```

**Explanation**:

* `monitor`: Metric to track
* `mode`: Maximize validation accuracy
* `save_top_k`: Save best model only

#### Early Stopping

```python
early_stopping = EarlyStopping(
    monitor="val_acc",
    mode="max",
    min_delta=1e-4,
    patience=5
)
```

**Explanation**:

* `min_delta`: Minimum improvement threshold
* `patience`: Stop after 5 non-improving epochs

#### Combined Callbacks

```python
callbacks = [model_checkpoint, early_stopping, ModelSummary(max_depth=1)]
```

---

### 9. Model Training & Evaluation

#### ResNet50 Training

```python
model = ResNet50()
trainer = L.Trainer(max_epochs=2, detect_anomaly=True, callbacks=callbacks)
trainer.fit(model, train_dataloader, val_dataloader)
```

**Test Results**:

* Test Accuracy: **0.5647**
* Test Loss: **1.1774**

---

#### ResNet101 Training

The key difference between **ResNet50** and **ResNet101** is the number of residual blocks in the **conv4_x** stage.

* ResNet50: 5 blocks
* ResNet101: **23 blocks**

```python
model_ResNet101 = ResNet101()
trainer = L.Trainer(max_epochs=6, detect_anomaly=True, callbacks=callbacks)
trainer.fit(model_ResNet101, train_dataloader, val_dataloader)
```

**Test Results**:

* Test Accuracy: **0.7312**
* Test Loss: **0.8384**

#### Comparison Between ResNet50 & ResNet101

After training, ResNet101 significantly outperforms ResNet50, achieving higher classification accuracy and lower test loss, demonstrating the benefit of increased network depth for 
feature representation and generalization.

---

## Conclusion

Through this project, I gained a solid understanding of how deep residual networks (ResNet) address the vanishing gradient problem and enable the training of very deep convolutional neural networks. 
I also learned how residual connections facilitate better gradient flow and improve model convergence compared to traditional deep CNNs.
On the practical side, my skills improved through building custom PyTorch datasets, designing data preprocessing pipelines, and managing experiments using PyTorch Lightning, including training loops, callbacks, 
and model checkpoints. 
Additionally, this project helped me better understand the importance of reproducibility, proper dataset splitting, and systematic evaluation when developing deep learning models.
