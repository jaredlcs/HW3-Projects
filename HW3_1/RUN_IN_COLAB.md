# Run in Google Colab - Quick Reference

## Method 1: One-Click Notebook (Easiest!)

1. Click this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaredlcs/HW3-Projects/blob/main/HW3_1/colab_notebook.ipynb)
2. In Colab, go to **Runtime → Change runtime type → Select "T4 GPU"**
3. Click **Runtime → Run all**
4. Wait ~15-30 minutes for training
5. Download `model.pth` and `train_plot.png` when complete

## Method 2: Copy-Paste Commands

Open a new Colab notebook and paste these cells:

### Cell 1: Setup
```python
# Install Box2D system dependencies
!apt-get update -qq
!apt-get install -y swig build-essential python3-dev

# Clone repository
!git clone https://github.com/jaredlcs/HW3-Projects.git
%cd HW3-Projects/HW3_1

# Install Python dependencies
!pip install -q gymnasium[box2d] torch matplotlib

import torch
print(f"✓ Setup complete | GPU: {torch.cuda.is_available()}")
```

### Cell 2: Train
```python
!python main.py
```

### Cell 3: View Results
```python
from IPython.display import Image, display
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.imshow(plt.imread('train_plot.png'))
plt.axis('off')
plt.show()
```

### Cell 4: Test
```python
!python main.py --test --no-render --test-episodes 20
```

### Cell 5: Download
```python
from google.colab import files
files.download('model.pth')
files.download('train_plot.png')
```

## If Session Times Out

Simply run:
```python
!python main.py --resume
```

## Repository

**GitHub:** https://github.com/jaredlcs/HW3-Projects

**Full Documentation:** See [README.md](README.md) and [COLAB_GUIDE.md](COLAB_GUIDE.md)
