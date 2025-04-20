# Summer_2025
# MNIST Two‑Layer Demo

Lightweight PyTorch example for digit classification.  
Purpose: prove I can build / train / evaluate a neural net before scaling to surgical‑video data.

---

## 1. Quick start

```bash
# clone
git clone https://github.com/hahnyu/Summer_2025.git
cd Summer_2025

# create environment
conda env create -f environment.yml
conda activate Summer_2025

# train + validate (≈ 30 s on CPU)
python mnist_practice.py
