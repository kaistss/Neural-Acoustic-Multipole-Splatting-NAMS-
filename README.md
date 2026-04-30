# 1. Create environment
conda create -n nams_env python=3.10 -y
conda activate nams_env

# 2. Install PyTorch
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 3. Install dependencies
pip install -r requirements.txt

# If tiny-cuda-nn installation fails, install it manually:
pip install git+https://github.com/NVlabs/tiny-cuda-nn/@d9e5274da3630ff747b67ed6188c838f4ad671ef#subdirectory=bindings/torch
