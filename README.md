# torch.rs

## Building

### Prerequisites

Install:

- CUDA (need headers)
- anaconda or miniconda
- `conda install pytorch torchvision cudatoolkit=YOUR_CUDA_VERSION -c pytorch` (you can install them in envs)

Then you have to set:

- CUDA_PATH
- TORCH_PATH

e.g.

```bash
export CUDA_PATH="/usr/local/cuda"
export TORCH_PATH=/home/sundoge/miniconda3/envs/pytorch1.0/lib/python3.6/site-packages/torch
```
