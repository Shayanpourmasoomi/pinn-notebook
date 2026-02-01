# Physics-Informed Neural Network (PINN) — Jupyter Notebook

This repository contains a Jupyter Notebook implementing a **Physics-Informed Neural Network (PINN)** to solve differential equations by combining neural networks with physical laws.

## File
- `pinn_Notebook.ipynb` — main notebook

## Overview
Physics-Informed Neural Networks (PINNs) are neural networks trained not only on data, but also constrained by the underlying **physical equations** (e.g. differential equations).  
Instead of relying purely on labeled data, the model minimizes a loss function that enforces the governing equations.

This notebook demonstrates:
- Defining a neural network for function approximation
- Enforcing physical constraints through the loss function
- Solving differential equations using PINNs

## Key Concepts
- Neural network approximation of continuous functions
- Automatic differentiation
- Physics-based loss functions
- Training with boundary and initial conditions

## Requirements
Python 3.9+ recommended.

Install dependencies:
```bash
pip install numpy matplotlib torch jupyter
