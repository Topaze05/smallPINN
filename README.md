# Deep Recursive Residual PINNs for Shallow Water Equations
**Course:** MATH 598 - Deep Learning (McGill University)

## Project Overview
This repository contains the implementation and research code for applying **Self-Recursion** and **Deep Supervision** mechanisms to Physics-Informed Neural Networks (PINNs).

The primary goal of this project is to emulate a deep residual network using a recursive architecture. Instead of stacking distinct layers to increase depth, this approach recycles a specific block of weights to iteratively refine latent representations. This methodology is tested on the **2D Shallow Water Equations (SWE)** using the **PDEBench** dataset.

### Key Features
*   **Novel Architecture (`AdaptedTRMPINN`):** A recursive network that iteratively refines latent features to approximate the solution, effectively emulating a much deeper network with fewer parameters.
*   **Fourier Features:** Integration of Fourier Feature mapping to mitigate spectral bias in standard MLPs.
*   **Masked Architectures:** Implementation of `MaskPINN` logic for adaptive activation.
*   **Two-Phase Training:** A robust training loop combining **Adam** (for coarse convergence) and **L-BFGS** (for fine-tuning).
*   **Physics-Informed Loss:** Solves the inverse problem by minimizing residuals of Mass and Momentum conservation laws alongside data loss.

## File Structure

```text
.
├── dataloader_.py        # Handles loading HDF5 files from PDEBench and data batching
├── demo.py               # Script to run a single demo training run and generate GIFs
├── models.py             # Neural Network Architectures (MLP, FF_PINN, MaskPINN, AdaptedTRMPINN)
├── PDE_Models.py         # Physics logic (SWE), Training loops, and Benchmarking scripts
├── config_comparison.json # Hyperparameters for comparing baseline models
└── README.md             # Project documentation
```

## Method: Self-Recursive Refinement

The core innovation lies in the `AdaptedTRMPINN` class (in `models.py`). Unlike a standard Feed-Forward network that passes data $x$ through layers $L_1 \dots L_N$, this model defines a refinement block $R$ and a number of recursive steps $T$.

$$ h_{t+1} = R(h_t + \text{Input}) $$

This allows the network to "think" deeper without increasing the parameter count, leveraging the residual connection $h_t$ to stabilize gradients during backpropagation through time.

## Installation & Requirements

1.  **Clone the repository.**
2.  **Install dependencies:**
    The project relies on PyTorch and standard scientific computing libraries.
    ```bash
    pip install torch numpy matplotlib h5py tqdm
    ```

## Dataset: PDEBench

This project uses the **2D Shallow Water Equations** dataset from [PDEBench](https://github.com/pdebench/PDEBench).

1.  Download the dataset (specifically the 2D SWE files: `2D_rdb_NA_NA.h5`).
2.  **Configuration:**
    You must update the `dataset_path` variable in `PDE_Models.py`, `dataloader_.py`, and `demo.py` to point to your local `2D_rdb_NA_NA.h5` file location.

    ```python
    # In demo.py and others:
    dataset_path = "2D_rdb_NA_NA.h5"
    ```

## Usage

### 1. Running a Demo
To train the Recursive PINN on a specific configuration and generate visualization GIFs (2D heatmap and 3D surface plot):

```bash
python demo.py
```
*   *Output:* This will save `results_demo.json` (loss logs) and `.gif` animations of the simulation comparison.

### 2. Running Benchmarks
To compare the proposed approach against standard MLPs, Fourier Feature networks, and Masked PINNs:

```bash
python PDE_Models.py
```
This script runs two functions:
1.  `comparison_new_approach`: Tests the recursive model with varying refinement blocks.
2.  `comparison_all_models`: Runs the baselines defined in `config_comparison.json`.

### 3. Training Logic
The model uses a composite loss function defined in `SWE_2D.compute_loss`:
$$ \mathcal{L} = \mathcal{L}_{data} + \lambda_{phys} \mathcal{L}_{PDE} + \lambda_{reg} \mathcal{L}_{reg} $$
*   **$\mathcal{L}_{data}$**: MSE or L1 loss against the ground truth simulation data.
*   **$\mathcal{L}_{PDE}$**: Residuals of the Mass and Momentum equations (Physics-Informed aspect).
*   **$\mathcal{L}_{reg}$**: Regularization on weights.

## Architectures Included

*   **`PINN_MLP`**: Standard fully connected network.
*   **`FF_PINN`**: MLP with Fourier Feature mapping for high-frequency coordinate embedding.
*   **`MaskPINN`**: Implements a masked layer architecture ($ \text{tanh}(z) \cdot (1 - e^{-\alpha z^2}) $) to enforce sparsity/modularity.
*   **`AdaptedTRMPINN`**: The proposed model using deep latent refinement blocks.

## Visualization

The `dataloader_.py` includes a helper `generate_comparison_gif` which produces side-by-side animations of the **Ground Truth** vs. **PINN Prediction**.

| 2D Heatmap Mode | 3D Surface Mode |
| :---: | :---: |
| Shows wave propagation intensity | Shows water height topology ($h$) |

## Author
**Jiajun Zhang** 
McGill Department of Mathematics and Statistics
jiajun.zhang@mail.mcgill.ca

**Emile Petit**
McGill Department of Mathematics and Statistics
emile.petit@mail.mcgill.ca


---
*Note: This code was developed for academic research purposes.*
