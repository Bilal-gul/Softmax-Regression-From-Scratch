# 🌲 Forest Cover Type Classification with Softmax Regression

<p align="center">
  <img src="https://github.com/Bilal-gul/Softmax-Regression-From-Scratch/blob/main/CoverType%20Prediction%20ai/images/loss_curve.png" width="750">
</p>

A complete implementation of **Softmax Regression (Multiclass Logistic Regression)** from scratch using **NumPy**. This project classifies forest cover types into one of seven classes without relying on machine learning libraries such as Scikit-Learn.

---

# Features

- Data preprocessing from scratch
- Dataset shuffling
- Train/Test split
- Z-Score feature normalization
- One-Hot Encoding
- Softmax activation
- Cross Entropy Loss
- Gradient Descent optimization
- Model parameter saving/loading
- Interactive prediction interface
- Confusion Matrix visualization
- Evaluation metrics implemented from scratch
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib

---

# Dataset

This project uses the **Forest CoverType Dataset**.

Due to GitHub file size considerations, the dataset is **not included** in this repository.

You can download it from one of the following sources:

- UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/31/covertype
  
---

# Project Structure

```text
Project/
│
├── images/
│   ├── loss_curve.png
│   └── confusion_matrix.png
│
├── model/
│   ├── softmax_regression_model.py # Core Softmax Regression class
│   ├── preprocessing.py            # Custom scaling, one-hot, and data splitting
│   └── metrics.py                  # Confusion matrix, Precision, Recall, F1 formulas
│           
├── train.py                        # Model training & evaluation pipeline
├── predict.py                      # Interactive terminal prediction CLI
└── README.md
```

---

# Usage

1. Download the Forest CoverType dataset.
2. Place the dataset inside the project directory.
3. Train the model:

```bash
python main.py
```

4. Make predictions:

```bash
python predict.py
```

---

# Mathematical Background

The Softmax Regression algorithm was implemented entirely from scratch using NumPy.

### Softmax Function

```text
P(y=i) = exp(z_i) / Σ exp(z_j)
```

### Cross Entropy Loss

```text
L = -(1/m) Σ Σ y(i,j) log(ŷ(i,j))
```

### Gradient

```text
dW = Xᵀ (Y - Ŷ) / m
```

### Weight Update

```text
W = W + α · dW
```

### Bias Gradient

```text
db = Σ (Y - Ŷ) / m
```

### Bias Update

```text
b = b + α · db
```

---

# Results

The model was evaluated using multiple classification metrics.

### Performance

- Accuracy
- Precision
- Recall
- F1 Score

Overall test accuracy is approximately **72%** using a manually implemented linear Softmax Regression model.

---

## Training Loss

The Cross Entropy Loss during training.

<p align="center">
  <img src="https://github.com/Bilal-gul/Softmax-Regression-From-Scratch/blob/main/CoverType%20Prediction%20ai/images/loss_curve.png" width="700">
</p>

---

## Confusion Matrix

Classification results on the test set.

<p align="center">
  <img src="https://github.com/Bilal-gul/Softmax-Regression-From-Scratch/blob/main/CoverType%20Prediction%20ai/images/confusion_matrix.png" width="700">
</p>

---

# What I Learned

During this project I implemented from scratch:

- Softmax Regression
- Cross Entropy Loss
- Gradient Descent
- One-Hot Encoding
- Feature Normalization
- Model Evaluation Metrics
- Confusion Matrix
- Object-Oriented Programming (OOP)
- Modular Project Design using Python

---

# Future Improvements

- L2 Regularization
- Mini-Batch Gradient Descent
- Learning Rate Scheduling
- Neural Networks
- Decision Tree and Random Forest comparison

---

# Requirements

```text
numpy
pandas
matplotlib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Author

Developed by **Bilal Gül**

This project was built for educational purposes to understand every mathematical component behind Softmax Regression by implementing the algorithm completely from scratch.
