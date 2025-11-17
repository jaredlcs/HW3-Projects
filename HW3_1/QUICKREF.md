# Quick Reference: Training Commands

## Local Training

### Start Fresh
```bash
python main.py
```

### Resume from Checkpoint
```bash
python main.py --resume
```

### Test Model
```bash
python main.py --test
```

### Test Without Rendering
```bash
python main.py --test --no-render
```

### Test Multiple Episodes
```bash
python main.py --test --test-episodes 20
```

## Google Colab

### Setup
```python
!pip install gymnasium[box2d] torch matplotlib
```

### Train
```python
!python main.py
```

### Resume
```python
!python main.py --resume
```

### Test
```python
!python main.py --test --no-render
```

### Download Files
```python
from google.colab import files
files.download('model.pth')
files.download('train_plot.png')
```

## File Outputs

After training:
- `model.pth` - Trained model weights
- `train_plot.png` - Learning curve
- `checkpoints/training_progress.pkl` - Resume checkpoint
- `checkpoints/metadata.json` - Training stats

## Key Hyperparameters

In `main.py`, see `Config` class:
- `LEARNING_RATE = 0.0003`
- `GAMMA = 0.99`
- `GAE_LAMBDA = 0.95`
- `CLIP_EPSILON = 0.2`
- `T_HORIZON = 2048`
- `K_EPOCHS = 4`
- `BATCH_SIZE = 64`
- `HIDDEN_DIM = 128`
- `TARGET_REWARD = 200`
- `MAX_EPISODES = 2000`

## Troubleshooting

### Installation
```bash
pip install torch gymnasium[box2d] numpy matplotlib
```

### Check Installation
```bash
python -c "import gymnasium; import torch; print('OK')"
```

### Missing Box2D
```bash
pip install box2d-py
```

## Expected Performance

- **Convergence:** 300-600 episodes (PPO is faster!)
- **Target Reward:** 200 (100-episode average)
- **Training Time:** 10-40 minutes
- **Success Rate:** >85% successful landings
