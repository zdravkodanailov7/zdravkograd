# ZdravkoGrad

A lightweight, scalar-valued autograd engine and neural network library built from scratch. It implements backpropagation over a dynamically built Directed Acyclic Graph (DAG) and features a PyTorch-like API.

![Decision Boundary Visualisation](/binary-classification-visualised.png)

## About

ZdravkoGrad is a pedagogical implementation of a deep learning framework. It breaks down the "black box" of tools like PyTorch by implementing the core mathematical machinery—gradients, the chain rule, and optimization—purely in Python.

It operates on scalar values (rather than large tensors) to make the mechanics of backpropagation transparent and easy to visualise.

**Key Features:**
* **Automatic Differentiation:** Implements reverse-mode autodiff (backpropagation) on a dynamically built computation graph.
* **PyTorch-like API:** Familiar `.backward()`, `.zero_grad()`, and `Module` based architecture.
* **Graph Visualisation:** Built-in tools to visualise the computation graph and gradients using Graphviz.
* **Neural Net Primitives:** Includes implementations of `Neuron`, `Layer`, and `MLP` classes.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zdravkodanailov7/zdravkograd.git
cd zdravkograd
````

2.  Install the required dependencies:
```bash
pip install numpy matplotlib graphviz scikit-learn
```

## Usage

### 1\. Basic Autograd

You can build complex mathematical expressions using the `Value` object. ZdravkoGrad tracks the operations and computes gradients automatically.

```python
from zdravkograd.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
f = Value(-2.0)
L = d * f

L.backward()

print(f'{L.data=}') # Output of the forward pass
print(f'{a.grad=}') # derivative of L with respect to a
print(f'{b.grad=}') # derivative of L with respect to b
```

### 2\. Training a Neural Network

You can define a Multi-Layer Perceptron (MLP) and train it on a dataset (e.g., `sklearn.datasets.make_moons`).

```python
from zdravkograd.nn import MLP

# Initialize a network: 2 input neurons, two hidden layers of 16, 1 output
model = MLP(2, [16, 16, 1])

# ... (data loading logic) ...

# Training Loop
for k in range(100):
    
    # Forward pass
    scores = list(map(model, inputs))
    
    # Calculate Loss (SVM Max-Margin + L2 Regularization)
    # ... (loss calculation logic) ...
    
    # Backward pass
    model.zero_grad()
    total_loss.backward()
    
    # Update weights (Stochastic Gradient Descent)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

*Check out `moons.ipynb` for the full training script and visualisation.*

## Project Structure

  * `zdravkograd/`: Source code for the library.
      * `engine.py`: Contains the `Value` class and autograd logic.
      * `nn.py`: Contains the `Neuron`, `Layer`, and `MLP` classes.
  * `learning.ipynb`: Introductory notebook explaining the calculus and building the engine step-by-step.
  * `moons.ipynb`: Demo notebook training a neural net on the non-linear "Moons" dataset.

## Acknowledgements

This project was heavily inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Building it was a comprehensive exercise in understanding the mathematical foundations of deep learning.

For a detailed walkthrough of how this was built, check out my [blog post](https://www.zdravkodanailov.com/projects/zdravkograd).

## License

MIT