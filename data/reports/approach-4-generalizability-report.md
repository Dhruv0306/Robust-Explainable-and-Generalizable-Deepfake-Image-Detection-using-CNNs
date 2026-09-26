## Approach 4 - Cross-manipulation generalizability

### Full-versus-LOO evaluation

The full-versus-LOO evaluation compares the baseline CNN detectors trained on all four FaceForensics++ manipulation categories with Leave-One-Manipulation-Out (LOO) detectors trained without one manipulation category. For each held-out category, the LOO model is evaluated on the corresponding held-out manipulation alongside the Real samples in the test set. Results are compared with the matching baseline model and random seed.

The evaluation covers four held-out manipulation categories, three CNN architectures (EfficientNet-B0, ResNet50, and Xception), and three random seeds (42, 123, and 2024). This gives 36 matched baseline-LOO comparisons. The reported results use video-level metrics. Values in Table X are the mean and standard deviation across the three seeds for each holdout-model combination.

The paired difference for a metric is calculated as:

**Delta M = LOO metric - baseline metric**

A negative difference indicates that the LOO model obtained a lower score than its matched baseline. The standard deviation of the paired differences describes how much the observed change varies across the three seeds.

### Results

**Table X. Video-level baseline and LOO performance by held-out manipulation and model.** Values are mean ± standard deviation across three seeds. Delta values are calculated as LOO minus baseline.

| Held-out manipulation | Model | Baseline F1 | LOO F1 | Δ F1 | Δ ROC-AUC |
|---|---|---:|---:|---:|---:|
| Deepfakes | EfficientNet-B0 | 0.834 ± 0.072 | 0.767 ± 0.088 | -0.067 ± 0.087 | -0.090 ± 0.070 |
| Deepfakes | ResNet50 | 0.906 ± 0.056 | 0.732 ± 0.135 | -0.174 ± 0.190 | -0.081 ± 0.040 |
| Deepfakes | Xception | 0.887 ± 0.061 | 0.691 ± 0.117 | -0.196 ± 0.137 | -0.037 ± 0.063 |
| Face2Face | EfficientNet-B0 | 0.863 ± 0.112 | 0.429 ± 0.132 | -0.435 ± 0.240 | -0.236 ± 0.079 |
| Face2Face | ResNet50 | 0.950 ± 0.056 | 0.784 ± 0.034 | -0.166 ± 0.024 | -0.102 ± 0.056 |
| Face2Face | Xception | 0.963 ± 0.064 | 0.782 ± 0.019 | -0.181 ± 0.068 | -0.120 ± 0.021 |
| FaceSwap | EfficientNet-B0 | 0.878 ± 0.088 | 0.268 ± 0.057 | -0.610 ± 0.054 | -0.428 ± 0.045 |
| FaceSwap | ResNet50 | 0.921 ± 0.034 | 0.147 ± 0.006 | -0.774 ± 0.039 | -0.495 ± 0.014 |
| FaceSwap | Xception | 0.963 ± 0.064 | 0.140 ± 0.134 | -0.823 ± 0.080 | -0.569 ± 0.054 |
| NeuralTextures | EfficientNet-B0 | 0.819 ± 0.090 | 0.396 ± 0.122 | -0.423 ± 0.182 | -0.287 ± 0.076 |
| NeuralTextures | ResNet50 | 0.872 ± 0.016 | 0.232 ± 0.078 | -0.640 ± 0.093 | -0.370 ± 0.053 |
| NeuralTextures | Xception | 0.904 ± 0.055 | 0.226 ± 0.074 | -0.678 ± 0.128 | -0.398 ± 0.069 |

### Observations

The LOO models obtained lower mean F1-scores than their matched baselines in all 12 holdout-model combinations. The size of the difference varied by held-out manipulation and architecture. For Deepfakes, the mean F1 reduction ranged from 0.067 for EfficientNet-B0 to 0.196 for Xception. For Face2Face, the reductions ranged from 0.166 to 0.435. The largest F1 reductions occurred when FaceSwap was held out, with decreases ranging from 0.610 to 0.823. NeuralTextures holdouts also showed substantial reductions, ranging from 0.423 to 0.678.

The ROC-AUC differences followed a similar pattern. The smallest mean ROC-AUC reduction was observed for Xception when Deepfakes was held out (-0.037). The largest reduction was observed for Xception when FaceSwap was held out (-0.569). For all three architectures, holding out FaceSwap produced the largest mean decrease in both F1-score and ROC-AUC among the four manipulation categories.

Seed-level variation differed across combinations. The standard deviation of the paired F1 differences was relatively high for Face2Face with EfficientNet-B0 (0.240), Deepfakes with ResNet50 (0.190), and NeuralTextures with EfficientNet-B0 (0.182). In comparison, the paired F1 difference for Face2Face with ResNet50 had a standard deviation of 0.024. These variations indicate that the mean change alone does not fully describe the results, and the seed-level values should be retained alongside the aggregated statistics.

Overall, the measured results show that performance on the held-out manipulation was lower for the LOO models than for the corresponding full-category baselines. The magnitude of this difference was not uniform across manipulation categories or architectures. Since each holdout-model combination was evaluated with three seeds, these results are reported descriptively; they do not establish statistical significance or identify the mechanism responsible for the observed performance differences.