<div align="center">

# Neural Network from Scratch

**A fully custom deep learning framework. No TensorFlow, no PyTorch, no Keras.**

**From the days before AI!**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Examples-Jupyter%20Notebook-orange?logo=jupyter)](Usage.ipynb)



</div>

---

## Overview

This project is a fully self-contained neural network framework written in pure Python and NumPy. Every component. From forward propagation to backpropagation, from SGD to Adam. Is implemented from first principles, without relying on any external machine learning library.

The primary goal is educational: to provide a transparent, readable, and modifiable foundation that reveals exactly what happens inside a neural network at every step.

---

## Features

| Category | Implementations |
|---|---|
| **Layers** | Dense (Fully Connected), Dropout |
| **Activations** | ReLU, Softmax, Sigmoid, Linear |
| **Optimizers** | SGD (+ momentum), Adagrad, RMSprop, Adam |
| **Loss Functions** | Categorical Crossentropy, Binary Crossentropy, MSE, MAE |
| **Regularization** | L1, L2 weight/bias penalties, Dropout |
| **Accuracy Metrics** | Categorical (binary & multi-class), Regression |
| **Callbacks** | Early Stopping with configurable patience |
| **Persistence** | Full model save/load, weights-only save/load |
| **Visualization** | Training curves, t-SNE dataset projection |
| **Dataset Tooling** | Image loading, histogram equalization, normalization, train/val/test split |

---

## Project Structure

```
neural_network_from_scratch/
│
├── lib/
│   ├── model.py          # Core Model class. Training, evaluation, prediction
│   ├── layers.py         # Dense and Dropout layer implementations
│   ├── activations.py    # ReLU, Softmax, Sigmoid, Linear
│   ├── optimizers.py     # SGD, Adagrad, RMSprop, Adam
│   ├── metrics.py        # Loss functions and accuracy metrics
│   ├── dataset.py        # Image dataset loader and preprocessor
│   ├── callbacks.py      # Early stopping callback
│   └── visualizers.py    # Training curve and t-SNE visualizations
│
├── Usage.ipynb           # End-to-end examples
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/szajbergyerek/neural_network_from_scratch.git
cd neural_network_from_scratch
pip install numpy opencv-python scikit-learn matplotlib
```

### Basic Usage

```python
from lib.model import Model
from lib.layers import Layer_Dense, Layer_Dropout
from lib.activations import Activation_ReLU, Activation_Softmax
from lib.optimizers import Optimizer_Adam
from lib.metrics import Loss_CategoricalCrossentropy, Accuracy_Categorical

# Build model
model = Model()
model.add(Layer_Dense(784, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(256, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Configure
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=5e-5),
    accuracy=Accuracy_Categorical()
)
model.finalize()

# Train
model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    early_stop=10
)

# Evaluate
model.evaluate(X_test, y_test)

# Visualize training
model.visualize_train()

# Save
model.save("my_model.pkl")
```

---

## Dataset Processing

The `ImageClassificationDataset` class provides a full preprocessing pipeline for image data:

```python
from lib.dataset import ImageClassificationDataset

dataset = ImageClassificationDataset()
dataset.load(path="./data", size=(64, 64))   # Loads images; subdirectory names = class labels
dataset.preprocess()                          # Histogram equalization (auto contrast/white balance)
dataset.normalize()                           # Scale pixel values to [-1, 1]
dataset.reshape()                             # Flatten 2D images to 1D vectors
dataset.visualize()                           # Preview 5x5 grid of samples
dataset.tsne()                                # t-SNE 2D projection of feature space

X_train, y_train, X_valid, y_valid, X_test, y_test = dataset.split(
    train=0.7, valid=0.2, test=0.1
)
```

---

## Architecture in Depth

### Forward Pass

```
Input → [Dense → Activation] × N → Output Activation → Predictions
```

### Backward Pass

Gradients are propagated in reverse order through all layers. When `Softmax` and `Categorical Crossentropy` are used together, the framework uses the combined analytical gradient for numerical efficiency.

### Optimizers

| Optimizer | Key Parameters |
|---|---|
| SGD | `learning_rate`, `momentum`, `decay` |
| Adagrad | `learning_rate`, `epsilon`, `decay` |
| RMSprop | `learning_rate`, `rho`, `epsilon`, `decay` |
| Adam | `learning_rate`, `beta_1`, `beta_2`, `epsilon`, `decay` |

### Regularization

L1 and L2 penalties can be applied independently to weights and biases on each `Layer_Dense`:

```python
Layer_Dense(
    n_inputs, n_neurons,
    weight_regularizer_l1=0.0,
    weight_regularizer_l2=5e-4,
    bias_regularizer_l1=0.0,
    bias_regularizer_l2=5e-4
)
```

---

## Model Persistence

```python
# Save full model (architecture + weights)
model.save("model.pkl")
model = Model.load("model.pkl")

# Save weights only (for transfer / fine-tuning)
model.save_parameters("weights.pkl")
model.load_parameters("weights.pkl")
```

---

## Examples

See [Usage.ipynb](Usage.ipynb) for complete end-to-end examples covering:
- Multi-class image classification
- Binary classification
- Regression
- Dataset visualization with t-SNE
- Model saving and loading

---

## References

- [Neural Networks from Scratch (nnfs.io)](https://nnfs.io/)
- [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
- [Neural Network Loss Functions](https://programmathically.com/an-introduction-to-neural-network-loss-functions/)
- [Optimization Algorithms for Neural Networks](https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html)
- [Deep Learning & Neural Networks](https://smltar.com/dldnn.html)

---

<div align="center">
Built from scratch. No magic. Just math.
</div>
