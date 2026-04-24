# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune itself during training using learnable gate parameters.

---

## Key Idea
Each weight has a gate (0–1). If the gate approaches 0, the weight is effectively removed from the network.

---

## Architecture
- Custom layer: PrunableLinear
- Dataset: CIFAR-10
- Loss = CrossEntropy + λ * L1(gates)

---

## Why L1 Regularization Encourages Sparsity
The L1 penalty adds a cost proportional to the magnitude of gate values. 
Since gates are constrained between 0 and 1 using a sigmoid function, 
minimizing the L1 loss pushes many gate values toward 0. 

This effectively disables the corresponding weights, resulting in a sparse network.

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.001  | 40.34%  | 0.16%    |
| 0.01   | XX%     | XX%      |
| 0.1    | XX%     | XX%      |

---

## Graphs

### Gate Distribution
![Gate Distribution](gate_lambda.png)

### Lambda vs Gate
![Lambda vs Gate](lambda_vs_gate.png)

### Sparsity vs Accuracy
![Sparsity vs Accuracy](sparsity_vs_accuracy.png)

---

## Conclusion
- Increasing lambda increases sparsity
- However, higher sparsity reduces model accuracy
- This demonstrates the trade-off between model compression and performance

---

## How to Run
```bash
pip install torch torchvision matplotlib
jupyter notebook
