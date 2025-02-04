# Student-Teacher Model for Image Anomaly Detection

This repository contains the **PyTorch implementation** of our paper published in the **IEEE Access journal**. You can read the full paper here: [IEEE Xplore](https://ieeexplore.ieee.org/document/10175536).

## 📌 Overview

This project implements a **Student-Teacher model** for anomaly detection in images. The method leverages a teacher network to guide a student network in learning normal patterns, allowing anomalies to be detected as deviations.

## 🚀 Getting Started

### **1. Prerequisites**

Ensure you have **Python 3.6** and **PyTorch 1.6.0** installed before running the code.

### **2. Install Dependencies**

Run the following command to install all required dependencies:

```sh
pip install -r requirements.txt
```

### **3. Prepare Datasets**

Download and place the required datasets in the `Datasets/` folder. Ensure the data structure matches the expected format.

### **4. Configure Hyperparameters**

Modify hyperparameters such as the **number of epochs**, **learning rate**, etc., in the `configs.yaml` file located in the `configs/` directory.

## 🔥 Training the Model

To train the model, use the following command:

```sh
python train.py
```

## 🎯 Testing the Model

To evaluate the model on test data, run:

```sh
python test.py
```

## 📂 Project Structure

```
Student-Teacher-Model-For-Image-Anomaly-Detection/
│── configs/           # Configuration files (configs.yaml)
│── datasets/          # Folder to store datasets
│── models/            # Model architecture definitions
│── utils/             # Helper functions and utilities
│── train.py           # Script for training the model
│── test.py            # Script for testing the model
│── requirements.txt   # List of dependencies
│── README.md          # Project documentation
```

## 🛠 Troubleshooting & Tips

- Ensure you are using the correct **Python** and **PyTorch** versions.
- If you encounter missing dependencies, manually install them using `pip install <package-name>`.
- Adjust hyperparameters in `configs.yaml` for better performance.

## 📜 Citation

If you find this work useful, please cite our paper:

```
@article{rakhmonov2023extensive,
  author    = {Rakhmonov, Akhrorjon Akhmadjon Ugli and Subramanian,     Barathi and Olimov, Bekhzod and Kim, Jeonghong},
  title     = {Extensive knowledge distillation model: An end-to-end effective anomaly detection model for real-time industrial applications},
  journal   = {IEEE Access},
  year      = {2023},
  doi       = {10.1109/ACCESS.2023.3293108}
}
```


