## Environment Setup

```bash
conda create -n NAMS python=3.10 -y
conda activate NAMS

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
