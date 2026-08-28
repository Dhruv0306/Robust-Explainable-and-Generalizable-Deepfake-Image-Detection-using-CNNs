## 🔬 Research Approach / Phase Reference
Select which Approach or Course Phase this PR addresses:
- [ ] **Approach 1 / Phase B:** CNN Baseline Detector
- [ ] **Approach 2 / Phase B:** Robustness under Image Transformations
- [ ] **Approach 3 / Phase C:** Explainability with Grad-CAM
- [ ] **Approach 4 / Phase C:** Generalizability to Unseen Manipulations
- [ ] **Phase D / Report:** Research / White Paper & Documentation
- [ ] **Chore / Setup / Other**

---

## 📝 Description
Provide a clear summary of the changes in this PR. What research objective does it address? Why was this solution chosen?

- 

## 🧪 Experimental Context & Hyperparameters (If applicable)
Details on experimental runs, models, or data changes:
* **Model Architecture:** (e.g., Xception, ResNet50, EfficientNet-B0)
* **Dataset Name & Split:** (e.g., FaceForensics++ c23, Subject-level split)
* **Hyperparameters Changed:** (e.g., learning rate, weight decay, epoch count)
* **Transformation details/Severity levels:** (e.g., JPEG quality 10-100, Gaussian noise variance)

## 📊 Results Summary & Metrics (If applicable)
If this PR includes experimental runs, summarize the key results (provide metrics table/comparison with baseline if relevant):

| Metric | Clean / Baseline | New / Transformed / OOD | Change ($\Delta$) |
|---|---|---|---|
| **Accuracy** | | | |
| **ROC-AUC** | | | |
| **F1-Score** | | | |
| **Precision** | | | |
| **Recall** | | | |

*Key Findings / Observations:*
- 

## 🔍 Code Changes & Quality Checks
Please verify:
- [ ] Preprocessing and evaluation protocols are identical to baseline (unless validating preprocessing changes)
- [ ] Random seed is set and documented for reproducibility (e.g., `seed=42`)
- [ ] No data leakage (subject or video level train-test leakage has been verified)
- [ ] Code follows style conventions (type annotations, structured docstrings)
- [ ] Performance regressions / errors have been checked
- [ ] Visualizations / Grad-CAM overlays conform to dataviz guidelines (if applicable)

## 📎 Checklist
- [ ] My code matches the existing style and conventions of the repository.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have updated the documentation or notebooks to reflect these changes.
- [ ] I have added/updated unit tests where necessary.
- [ ] All tests pass locally.
- [ ] Commit messages follow the project convention (`type(scope): subject`).

<!--
🤖 Generated with Claude Code
-->
