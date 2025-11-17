# Google Colab Quick Start Guide
# HW3-1: LunarLander-v3 Training

This guide helps you train the LunarLander agent on Google Colab.

## Quick Start (Copy-Paste into Colab)

### Step 1: Install Dependencies

```python
# Run this cell first
!pip install gymnasium[box2d] torch matplotlib
```

### Step 2: Upload main.py

Use Colab's file upload feature:
1. Click the folder icon on the left sidebar
2. Click the upload button
3. Upload `main.py` from your local machine

Or clone from GitHub if you have your code there:
```python
# !git clone https://github.com/yourusername/your-repo.git
# %cd your-repo/HW3-Projects/HW3_1
```

### Step 3: Start Training

```python
# Train from scratch
!python main.py
```

### Step 4: If Session Times Out

If your Colab session times out, simply:

```python
# Resume from checkpoint
!python main.py --resume
```

All your progress is saved!

### Step 5: Test Your Model

```python
# Test the trained model (no rendering on Colab)
!python main.py --test --no-render
```

### Step 6: Download Results

```python
# Download model and plot
from google.colab import files

files.download('model.pth')
files.download('train_plot.png')
```

## Full Colab Notebook Cell-by-Cell

### Cell 1: Setup

```python
# Install dependencies
!pip install -q gymnasium[box2d] torch matplotlib

print("✓ Dependencies installed")
```

### Cell 2: Upload Code

```python
# Upload main.py
from google.colab import files
uploaded = files.upload()
print("✓ Files uploaded")
```

### Cell 3: Training

```python
# Train the agent
!python main.py

# If resuming from checkpoint:
# !python main.py --resume
```

### Cell 4: View Training Plot

```python
# Display the training curve
from IPython.display import Image, display
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
img = plt.imread('train_plot.png')
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
```

### Cell 5: Test Model

```python
# Test the trained model
!python main.py --test --no-render --test-episodes 20
```

### Cell 6: Download Results

```python
# Download all results
from google.colab import files

files.download('model.pth')
files.download('train_plot.png')

# Also download checkpoint (optional, for resuming)
!zip -r checkpoints.zip checkpoints/
files.download('checkpoints.zip')

print("✓ All files downloaded")
```

## Monitoring Training

### Check GPU Status

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Monitor Progress

Training will print progress every 10 episodes:
```
Episode 10/2000 | Score: 45.23 | Avg Score (100): -123.45
Episode 20/2000 | Score: 78.91 | Avg Score (100): -98.76
...
```

### Expected Training Time

- **With GPU (T4):** ~15-30 minutes
- **Without GPU:** ~30-60 minutes
- **Convergence:** Usually 500-1000 episodes

## Tips for Colab

### 1. Keep Session Alive

Colab sessions timeout after ~90 minutes of inactivity. To prevent this:
- Keep the browser tab active
- Use a browser extension like "Colab Auto-Reconnect"
- Or just resume training with `--resume` flag

### 2. Save Checkpoints Frequently

The code automatically saves checkpoints every 50 episodes to:
```
checkpoints/training_progress.pkl
checkpoints/metadata.json
```

### 3. Download Checkpoints Periodically

```python
# Every few hundred episodes, download checkpoint as backup
!zip -r checkpoints_backup.zip checkpoints/
from google.colab import files
files.download('checkpoints_backup.zip')
```

### 4. Use GPU Runtime

1. Go to Runtime → Change runtime type
2. Select "T4 GPU" or "GPU"
3. Click Save

Training will automatically use GPU if available.

### 5. Mount Google Drive (Optional)

Save checkpoints directly to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Modify checkpoint directory in main.py or copy files after training
!cp -r checkpoints/ /content/drive/MyDrive/HW3_checkpoints/
!cp model.pth /content/drive/MyDrive/
!cp train_plot.png /content/drive/MyDrive/
```

## Troubleshooting

### Box2D Import Error

```python
!apt-get install -y swig
!pip install gymnasium[box2d]
```

### Out of Memory

If you run out of memory:
- Use CPU runtime (GPU has more memory)
- Reduce batch size (not applicable for A2C, but keep in mind)
- Restart runtime and try again

### Training Not Converging

If training isn't reaching 200 average reward:
- Let it run longer (try 2000-3000 episodes)
- Check the learning curve for upward trend
- Make sure checkpoints are loading correctly

## Complete Colab Notebook Template

Here's a complete single notebook you can copy:

```python
# ============================================================
# HW3-1: LunarLander Training on Google Colab
# ============================================================

# Cell 1: Setup
!pip install -q gymnasium[box2d] torch matplotlib
import torch
print(f"✓ Setup complete | GPU: {torch.cuda.is_available()}")

# Cell 2: Upload main.py
from google.colab import files
uploaded = files.upload()

# Cell 3: Train (or resume)
!python main.py
# !python main.py --resume  # Uncomment to resume

# Cell 4: View results
from IPython.display import Image, display
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.imshow(plt.imread('train_plot.png'))
plt.axis('off')
plt.show()

# Cell 5: Test model
!python main.py --test --no-render --test-episodes 20

# Cell 6: Download
files.download('model.pth')
files.download('train_plot.png')
print("✓ Downloads complete")
```

## Next Steps

After training:
1. Download `model.pth` and `train_plot.png`
2. Test locally with rendering: `python main.py --test`
3. Record a 1-minute demo video
4. Upload video to YouTube
5. Add link to README.md

## Resources

- [Google Colab Docs](https://colab.research.google.com/)
- [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [PyTorch on Colab](https://colab.research.google.com/notebooks/gpu.ipynb)

---

Good luck with your training! 🚀
