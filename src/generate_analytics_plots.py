"""
Generate comparative data analytics figures across models and seeds for Approach 1 report.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dir = Path("data/reports/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Collect data
output_dir = Path("data/output")
runs = []
for run_dir in sorted(output_dir.iterdir()):
    if run_dir.is_dir():
        cfg_file = run_dir / "config.json"
        res_file = run_dir / "test_results.json"
        if cfg_file.exists() and res_file.exists():
            cfg = json.loads(cfg_file.read_text())
            res = json.loads(res_file.read_text())
            runs.append({
                "model": cfg.get("model"),
                "seed": cfg.get("seed"),
                "metrics": res.get("video_metrics", {}).get("mean", {})
            })

models = ["xception", "efficientnet_b0", "resnet50"]
seeds = [42, 123, 2024]

# Organize metrics
data = {m: {s: None for s in seeds} for m in models}
for r in runs:
    data[r["model"]][r["seed"]] = r["metrics"]

# 1. Plot 1: Accuracy & F1-Score Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
x = np.arange(len(models))
width = 0.25

accs = [[data[m][s]["accuracy"] for s in seeds] for m in models]
f1s = [[data[m][s]["f1"] for s in seeds] for m in models]

acc_means = [np.mean([data[m][s]["accuracy"] for s in seeds]) for m in models]
f1_means = [np.mean([data[m][s]["f1"] for s in seeds]) for m in models]
acc_stds = [np.std([data[m][s]["accuracy"] for s in seeds]) for m in models]
f1_stds = [np.std([data[m][s]["f1"] for s in seeds]) for m in models]

rects1 = ax.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Accuracy (Mean ± Std)', capsize=5, color='#2b5c8f', alpha=0.85)
rects2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds, label='F1-Score (Mean ± Std)', capsize=5, color='#d95f02', alpha=0.85)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Approach 1: Video-Level Accuracy and F1-Score Across Backbones (3 Seeds)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Xception', 'EfficientNet-B0', 'ResNet50'], fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor='white', fontsize=11)
ax.set_ylim(0.7, 1.05)

# Add values on bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig_path_1 = fig_dir / "model_comparison_accuracy_f1.png"
plt.savefig(fig_path_1)
plt.close()
print(f"Saved {fig_path_1}")

# 2. Plot 2: ROC-AUC Comparison across Models and Seeds
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
aucs = [[data[m][s]["roc_auc"] for s in seeds] for m in models]
auc_means = [np.mean(a) for a in aucs]
auc_stds = [np.std(a) for a in aucs]

colors = ['#2b5c8f', '#7570b3', '#d95f02']
bars = ax.bar(['Xception', 'EfficientNet-B0', 'ResNet50'], auc_means, yerr=auc_stds, capsize=5, color=colors, width=0.5, alpha=0.85)

ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
ax.set_title('Approach 1: Video-Level ROC-AUC Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0.85, 1.03)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig_path_2 = fig_dir / "model_comparison_roc_auc.png"
plt.savefig(fig_path_2)
plt.close()
print(f"Saved {fig_path_2}")

# 3. Plot 3: Seed Stability / Variance Comparison (Box/Scatter)
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
seed_data = [[data[m][s]["accuracy"] for s in seeds] for m in models]
bp = ax.boxplot(seed_data, tick_labels=['Xception', 'EfficientNet-B0', 'ResNet50'], patch_artist=True,
                boxprops=dict(facecolor='#a6cee3', color='#1f78b4', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='#1f78b4', linewidth=1.5),
                capprops=dict(color='#1f78b4', linewidth=1.5))

for i, m_data in enumerate(seed_data):
    y = m_data
    x = np.random.normal(i + 1, 0.04, size=len(y))
    ax.plot(x, y, 'ko', alpha=0.8, markersize=8)

ax.set_ylabel('Video Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Approach 1: Seed Sensitivity & Distribution (N=3 Seeds per Model)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0.8, 1.05)

plt.tight_layout()
fig_path_3 = fig_dir / "seed_sensitivity_distribution.png"
plt.savefig(fig_path_3)
plt.close()
print(f"Saved {fig_path_3}")
