## Environment Setup

This code was tested with the following environment:

- Ubuntu Linux
- Python 3.10
- PyTorch 2.7.1
- CUDA 12.6
- NVIDIA GPU
- tiny-cuda-nn

We recommend using a separate conda environment.

### Installation

```bash
conda create -n nams_env python=3.10 -y
conda activate nams_env

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
