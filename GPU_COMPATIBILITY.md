# GPU Compatibility Guide

## PyTorch: CPU and GPU Support

PyTorch automatically detects and uses available hardware. The same package works for both CPU and GPU.

### Installation

```bash
# One command works for both CPU and GPU machines
pip install torch torchvision
```

PyTorch will:
- Use CUDA GPU if available
- Fall back to CPU if no GPU detected
- No separate installation needed

### Verify GPU Detection

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")
```

### Code Compatibility

Our implementation automatically handles both:

```python
# From src/utils.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

No changes needed when switching machines.

## Face Detection: MTCNN

`facenet-pytorch` (MTCNN) automatically uses GPU when available:

```python
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
```

Same code runs on CPU or GPU.

## Performance Notes

**GPU machines:**
- Training: 5-10x faster
- Face detection: 3-5x faster
- Recommended batch size: 32-64

**CPU machines:**
- Training: slower but works
- Face detection: functional, takes longer
- Recommended batch size: 8-16

Batch size auto-adjusts via `src/utils.py:get_batch_size()`.

## Mixed Environment Setup

Install once, run everywhere:

```bash
# Install on machine with GPU
pip install -r requirements.txt

# Copy .venv to CPU machine
# Works without reinstall
```

PyTorch binary includes both CPU and CUDA code. Device selection happens at runtime.
