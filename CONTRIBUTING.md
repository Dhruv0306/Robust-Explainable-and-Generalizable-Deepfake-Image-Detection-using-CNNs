# Contributing to Robust, Explainable, and Generalizable Deepfake Image Detection using CNNs

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing to the codebase.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Conventions](#commit-conventions)
- [Branching Strategy](#branching-strategy)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Experimental Reproducibility](#experimental-reproducibility)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

---

## Project Overview

This is an empirical research project for Advanced Data Mining (CS G520) at BITS Pilani, Goa Campus. The project investigates robustness, explainability, and generalizability of CNN-based deepfake image detectors across four cumulative approaches:

1. **Approach 1:** CNN Baseline Detector
2. **Approach 2:** Robustness under Image Transformations
3. **Approach 3:** Explainability with Grad-CAM
4. **Approach 4:** Generalizability to Unseen Manipulations

Contributions should align with the research objectives and maintain experimental integrity.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Access to FaceForensics++ dataset (primary)
- GPU with CUDA support (recommended for training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/Robust-Explainable-and-Generalizable-Deepfake-Image-Detection-using-CNNs.git
   cd Robust-Explainable-and-Generalizable-Deepfake-Image-Detection-using-CNNs
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Git commit template:**
   ```bash
   git config commit.template .gitmessage
   ```

---

## Development Workflow

1. **Create a feature branch** from `main`
2. **Make your changes** following code style guidelines
3. **Test your changes** locally
4. **Commit with conventional commit messages**
5. **Push to your branch**
6. **Open a Pull Request** using the PR template

---

## Commit Conventions

We follow **Conventional Commits** with project-specific types and scopes.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat` — New feature or functionality
- `fix` — Bug fix
- `data` — Dataset processing, splits, or data pipeline changes
- `model` — Model architecture, training, or checkpoint changes
- `experiment` — Experiment configuration, execution, or results
- `eval` — Evaluation metrics, scripts, or analysis
- `viz` — Visualizations, Grad-CAM, plots, or figures
- `docs` — Documentation changes (README, reports, comments)
- `refactor` — Code restructuring without changing behavior
- `test` — Adding or updating tests
- `config` — Configuration files, hyperparameters, environment
- `ci` — CI/CD pipeline changes
- `chore` — Maintenance tasks (dependencies, .gitignore, cleanup)
- `style` — Code formatting (no logic changes)

### Scopes (optional)

- `baseline` — Approach 1: CNN baseline detector
- `robustness` — Approach 2: image transformation experiments
- `explainability` — Approach 3: Grad-CAM and XAI analysis
- `generalizability` — Approach 4: cross-manipulation/cross-dataset
- `data` — Data loading, preprocessing, augmentation
- `training` — Training loop, optimizer, scheduler
- `evaluation` — Metrics computation, result aggregation
- `notebook` — Jupyter notebooks
- `report` — Research paper, white paper, reports

### Examples

```
feat(baseline): add Xception model with ImageNet pretrained weights

model(robustness): add Gaussian blur at severity levels 1-5

eval(generalizability): compute cross-dataset gap on Celeb-DF v2

viz(explainability): generate Grad-CAM overlays for false negatives

fix(training): resolve learning rate scheduler reset on resume
```

### Commit Message Guidelines

- Use imperative mood in subject line ("add" not "added" or "adds")
- Do not capitalize the first letter of the subject
- No period at the end of the subject line
- Wrap body at 72 characters
- Use body to explain WHAT and WHY, not HOW
- Reference issues in footer: `Closes #123`, `Refs #456`

---

## Branching Strategy

### Main Branches

- `main` — Stable codebase with reviewed, tested code

### Feature Branches

Use descriptive branch names following this pattern:

```
<type>/<short-description>
```

**Examples:**
- `feat/add-xception-baseline`
- `experiment/robustness-jpeg-compression`
- `viz/gradcam-false-negatives`
- `fix/data-leakage-split`
- `docs/update-readme-approach-2`

### Branch Lifecycle

1. **Create** from `main`
2. **Develop** with regular commits
3. **Push** to remote
4. **Open PR** when ready for review
5. **Merge** after approval
6. **Delete** feature branch after merge

---

## Pull Request Process

1. **Fill out the PR template** completely:
   - Select the research approach/phase
   - Provide clear description
   - Document experimental context (if applicable)
   - Summarize results and metrics (if applicable)
   - Complete all checklists

2. **Ensure CI passes** (when CI is configured)

3. **Request review** from team members

4. **Address feedback** promptly

5. **Squash commits** if requested before merging

6. **Reference related issues** using keywords:
   - `Closes #123` — Automatically closes issue when PR is merged
   - `Refs #456` — Links to issue without closing

---

## Code Style Guidelines

### Python

- **PEP 8** compliance
- **Type annotations** for function signatures
- **Docstrings** for all public functions and classes (Google style preferred)
- **Maximum line length:** 100 characters
- **Imports:** Organized (standard library, third-party, local)

**Example:**

```python
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from src.models.base import BaseDetector


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    correct = (predictions.argmax(dim=1) == labels).sum().item()
    total = labels.size(0)
    return correct / total
```

### Jupyter Notebooks

- **Clear markdown headers** for sections
- **Comments** explaining complex logic
- **Output cells cleared** before committing (unless documenting results)
- **Consistent naming** for variables across notebooks

### File Naming

- Python modules: `lowercase_with_underscores.py`
- Notebooks: `descriptive_name_approach_X.ipynb`
- Config files: `config_baseline.yaml`, `config_robustness_jpeg.yaml`
- Result files: `results_approach_2_blur_severity_3.json`

---

## Experimental Reproducibility

Maintaining reproducibility is critical for research integrity.

### Requirements

1. **Set random seeds:**
   ```python
   import random
   import numpy as np
   import torch

   def set_seed(seed: int = 42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
   ```

2. **Document hyperparameters:**
   - Use configuration files (YAML/JSON)
   - Include all hyperparameters in experiment logs
   - Save configuration with model checkpoints

3. **Log experimental settings:**
   - Model architecture
   - Dataset version and split
   - Preprocessing steps
   - Training configuration
   - Hardware used

4. **Avoid data leakage:**
   - Use video-level or subject-level splits where possible
   - Document split strategy clearly
   - Never use test set for validation

5. **Version datasets:**
   - Document dataset version (e.g., FaceForensics++ c23)
   - Track preprocessing transformations
   - Store data split indices

---

## Testing

### Types of Tests

1. **Unit tests** — Test individual functions and modules
2. **Integration tests** — Test data pipelines and training loops
3. **Validation tests** — Verify evaluation metrics computation

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

```python
import pytest
import torch
from src.evaluation.metrics import compute_roc_auc


def test_compute_roc_auc_perfect_classifier():
    """Test ROC-AUC for perfect classifier."""
    predictions = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    
    auc = compute_roc_auc(predictions, labels)
    
    assert auc == 1.0, "Perfect classifier should have AUC = 1.0"
```

---

## Documentation

### Code Documentation

- **Docstrings** for all public APIs
- **Inline comments** for complex logic
- **README files** in each major directory

### Experimental Documentation

- **Notebooks** should be self-contained with explanations
- **Results** should be documented in `results/` directory
- **Reports** should reference experimental configurations

### Updating Documentation

- Update README when adding new features or approaches
- Document changes in notebooks when modifying experiments
- Add entries to research reports for completed experiments

---

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for answers
3. **Verify the issue** is reproducible

### Creating an Issue

Use the appropriate issue template:

- **Bug Report** — For reporting bugs or unexpected behavior
- **Feature Request** — For proposing new features or enhancements

### Issue Guidelines

- **Clear, descriptive title**
- **Complete template fields**
- **Minimal reproducible example** for bugs
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU)

---

## Team

| Name | Role |
|---|---|
| **Dhruv Rakeshbhai Patel** | Project Group Member |
| **Kunta Vikranth Reddy** | Project Group Member |
| **Saketh Sridharan** | Project Group Member |
| **Nitish Katteboyina** | Project Group Member |

**Course:** Advanced Data Mining (CS G520)  
**Instructor-in-Charge:** Dr. Hemant Rathore  
**Institution:** BITS Pilani, Goa Campus

---

## Questions?

If you have questions about contributing, please:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the `question` label

---

## License

This project is released under the [MIT License](LICENSE).

---

**Thank you for contributing to deepfake detection research!**
