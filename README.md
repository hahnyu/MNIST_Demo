# MNIST Two‑Layer Demo

Lightweight PyTorch example for digit classification.  
Purpose: prove I can build / train / evaluate a neural net before scaling to surgical‑video data.

---

## Start

```bash
# clone
git clone https://github.com/hahnyu/MNIST_Demo.git
cd MNIST_Demo

# create environment
conda env create -f environment.yml
conda activate MNIST_Demo

# train + validate (≈ 30 s on CPU)
python mnist_practice.py
```
Outputs mnist_two_layer.pth and confusion_matrix.png.  

## Model
Input: 28×28 grayscale image → flattened (784 dims)  
Hidden layer: 128 ReLU units  
Output layer: 10 logits (digits 0–9)  
Parameters: ~101 k  
Confusion matrix:  
![Confusion Matrix](confusion_matrix.png)

## Results
Validation accuracy (epoch 5): 0.950  
Parameters: 101,770  
Inference latency (GPU, batch 128): 0.01 ms/image  

## Credits
Created by Hamilton Jeong (Cornell Biometry ’28) as a warm‑up for surgical AI research.
