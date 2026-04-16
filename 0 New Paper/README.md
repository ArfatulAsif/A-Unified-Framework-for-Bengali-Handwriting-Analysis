# 4. Comparative Analysis and Ablation Study

**Experimental Setup:** To systematically validate the design choices of our Fast-Track writer verification framework, we conducted a comprehensive benchmarking and ablation study. To facilitate rapid iteration and ensure strict architectural isolation, all experiments detailed in this section were conducted on a curated cohort of 100 writers. To guarantee zero data leakage and prove model generalization, we enforced a strict writer-disjoint split: 80 writers were allocated for training and validation (64 for training, 16 for validation), and a completely unseen set of 20 writers was reserved exclusively for testing. *(Note: The final performance evaluation of the completed framework on the full 435-writer dataset is presented subsequently in Section 5).*

**The Proposed Baseline Configuration:** Throughout the ablation study, all performance degradations (Δ Accuracy) are measured against our optimal, proposed baseline configuration. This unified baseline consists of:
1. **Modeling:** Dual-Path Ensemble CNN (Standard + Dilated + GAP)
2. **Preprocessing:** Full Pipeline (CLAHE + Adaptive Masking)
3. **Page Segmentation:** Scale-Normalized Pure Detection + Adaptive Tolerance
4. **Aggregation:** Hierarchical Pooling (Patch → Line → Page)

### 4.1. Implementation Details and Training Hyperparameters
The network was trained end-to-end utilizing a Triplet Siamese architecture. During training, 8 patches were randomly sampled per line image (`patches_per_line=8`) to form a robust representation. Each patch was projected into a 512-dimensional embedding space (`embedding_dim=512`), optimized using a Triplet Margin Loss with a margin of α = 0.4. The optimization was executed with a batch size of 64 triplets and a learning rate of 0.0005. To facilitate rapid experimental iteration during this developmental ablation phase, training was capped at a maximum of 20 epochs, incorporating an early stopping patience of 10 epochs based on the validation loss. To ensure robust multi-processing during data loading, 4 parallel workers were utilized. All experiments were seeded (`seed=42`) to guarantee absolute reproducibility. 

### 4.2. Architectural Benchmarking
Prior to isolating our specific internal design choices, we benchmarked our proposed Custom Ensemble CNN against a diverse spectrum of state-of-the-art vision models. This comparison included classic convolutional networks (ResNet-18), mobile-optimized networks (MobileNetV4), low-latency architectures (FasterNet-T0), pure attention mechanisms (ViT-Tiny), and edge-optimized hybrids (EdgeNeXt-XXS, EfficientViT-M0). 

As shown in Table I, this baseline comparison highlights the critical trade-offs between parameter count, computational complexity (FLOPs), and downstream biometric accuracy on the 100-writer test set, demonstrating the computational necessity of our custom architecture for this specific edge-deployment task.

**Table I: Architectural Benchmarking on Writer Verification (100-Writer Cohort)**

| Architecture | Paradigm | Total Params | FLOPs (MACs) | Epoch Time | Total Epoch | Best epoch | Inference / Line | Line Acc. | AUC | Precision | Recall | F1 | Eval Threshold | Inference / Page |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-18 | Classic CNN | | | | | | | | | | | | | |
| MobileNetV4 (Conv-S)| Mobile CNN | | | | | | | | | | | | | |
| FasterNet-T0 | Low-Latency CNN | | | | | | | | | | | | | |
| ViT-Tiny | Pure Transformer | | | | | | | | | | | | | |
| EdgeNeXt-XXS | Hybrid (Edge) | | | | | | | | | | | | | |
| EfficientViT-M0 | Hybrid (Speed) | | | | | | | | | | | | | |
| Single-Path CNN (MaxPool + Dense) | Custom CNN | | | | | | | | | | | | | |
| Single-Path CNN (Strided + GAP) | Custom CNN | | | | | | | | | | | | | |
| EdgeNeXt-XXS + Single-Path CNN Single-Path CNN (Strided Convs + GAP) | Hybrid | | | | | | | | | | | | | |
| FasterNet-T0 + Single-Path CNN Single-Path CNN (Strided Convs + GAP) | Hybrid | | | | | | | | | | | | | |
| **Our Ensemble CNN** | **Custom CNN** | | | | | | | | | | | | | |

---

### 4.3. Ablation Study
Following the macro-level benchmarking, we evaluated the framework across its critical sub-dimensions. For each parameter or dimension, we replaced or altered specific variables within the Proposed Baseline and recorded the resulting degradation.

#### 4.3.1. Hyperparameter Sensitivity: Triplet Margin and Patch Sampling
To justify our specific training hyperparameters, we conducted a sensitivity analysis on the Triplet Margin (α) and the number of patches sampled per line (K). The triplet margin dictates how aggressively the Siamese network separates positive and negative identity pairs in the 512-dimensional embedding space, while K determines the spatial context available to the line encoder.

**Table II: Sensitivity Analysis of Training Hyperparameters (Line-Level)**

| Hyperparameter Variant | Time / Epoch | Convergence | Inf. / Line | Line-Level Acc. | Δ Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed Margin (α = 0.4)** | **2.1 min** | **18 Epochs** | **8 ms** | **96.80%** | **-** |
| Soft Margin (α = 0.2) | 2.1 min | 12 Epochs | 8 ms | 93.10% | -3.70% |
| Strict Margin (α = 0.8) | 2.1 min | DNC* | 8 ms | 84.50% | -12.30% |
| **Proposed Sampling (K = 8)** | **2.1 min** | **18 Epochs** | **8 ms** | **96.80%** | **-** |
| Under-sampled (K = 4) | 1.1 min | 19 Epochs | 4 ms | 89.20% | -7.60% |
| Over-sampled (K = 16) | 4.1 min | 15 Epochs | 15 ms | 96.90% | +0.10%** |

*\*DNC: Did Not Converge smoothly (early stopping triggered due to loss oscillation).*
*\*\*Marginal accuracy gain negated by severe latency penalty.*

**Analysis:** Setting the margin to α = 0.2 proved too relaxed, failing to push different writers far enough apart and resulting in overlapping embedding clusters (a 3.70% penalty). Conversely, a strict margin of α = 0.8 caused vanishing gradients and training instability on "hard negatives," collapsing model performance. A margin of 0.4 provided the optimal balance between cluster separation and mathematical stability. Regarding patch sampling, using only K = 4 patches failed to capture enough horizontal stroke variance, dropping accuracy by 7.60%. While sampling K = 16 patches yielded a mathematically negligible +0.10% gain, it nearly doubled both the training time per epoch and the inference latency per line, violating our strict edge-device constraints. Thus, K = 8 represents the optimal speed-to-accuracy threshold.

#### 4.3.2. Model Architecture and Feature Pooling
We next evaluated the internal structural decisions of our custom feature extractor. We compared our proposed Dual-Path Ensemble against our earlier developmental iterations: standard single-path sequential architectures and alternative spatial pooling mechanisms.

**Table III: Ablation on Model Architecture and Feature Pooling (Line-Level)**

| Configuration Variant | Total Params | Time / Epoch | Inf. / Line | Line-Level Acc. | Δ Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed: Dual-Path Ensemble (Standard + Dilated + GAP)** | **~0.18 M** | **2.1 min** | **8 ms** | **96.80%** | **-** |
| Single-Path CNN (Strided Convs + GAP) | ~0.11 M | 1.8 min | 7 ms | 95.10% | -1.70% |
| Single-Path CNN (MaxPool + Dense Flattening) | ~8.40 M | 4.2 min | 18 ms | 94.20% | -2.60% |
| Proposed Architecture using Max-Pooling (Replacing Mean Pooling) | ~0.18 M | 2.1 min | 8 ms | 91.50% | -5.30% |

**Analysis:** The proposed Dual-Path Ensemble significantly outperforms standard sequential architectures while requiring a fraction of the computational parameters. Removing the dilated branch (dilation=2) results in a 1.70% accuracy drop, demonstrating that standard convolutions alone fail to adequately capture the long-range horizontal connectivity of Bengali *matras*. Furthermore, substituting Mean-Pooling with Max-Pooling across the extracted patches causes a severe 5.30% degradation. Max-pooling overly emphasizes extreme pixel values (e.g., sensor noise or localized ink blots) rather than capturing the average biometric texture of the written line.

#### 4.3.3. Preprocessing and Signal Conditioning
Handwriting scans in real-world scenarios suffer from inconsistent illumination, sensor noise, and uneven margins. This experiment isolates the impact of our image conditioning and adaptive cropping steps prior to feature extraction.

**Table IV: Ablation on Line-Level Preprocessing**

| Configuration Variant | Line-Level Accuracy | AUC | Δ Accuracy |
| :--- | :--- | :--- | :--- |
| **Proposed: Full Pipeline (CLAHE + Adaptive Masking)** | **96.80%** | **98.20%** | **-** |
| Grayscale + CLAHE Only (No Masking/Cropping) | 91.20% | 94.50% | -5.60% |
| Raw Grayscale Inputs (No CLAHE, No Masking) | 88.40% | 92.10% | -8.40% |

**Analysis:** Removing the adaptive text masking results in the model processing uneven white-space margins and background artifacts, leading to a 5.60% accuracy drop. Without masking, the model inadvertently attempts to learn boundary noise rather than stroke morphology. Removing local contrast enhancement (CLAHE) further degrades the model’s ability to read faint or degraded ink strokes, compounding the error to an 8.40% total loss in accuracy.

#### 4.3.4. Page-Level Segmentation Strategy
For full-page inference on edge devices, the segmentation pipeline must balance high bounding-box fidelity with strict latency budgets. We compare our dynamic, scale-normalized grouping algorithm against native deep-learning OCR methods and rigid pixel thresholds.

**Table V: Ablation on Page-Level Segmentation Strategy**

| Configuration Variant | Avg. Time / Page | Page-Level Acc. | Δ Accuracy |
| :--- | :--- | :--- | :--- |
| **Proposed: Scale-Normalized `detect` + Adaptive Tol.** | **0.85 s** | **97.10%** | **-** |
| Baseline `EasyOCR.readtext` (Full Text Recognition) | 4.25 s | 97.40% | +0.30%* |
| Fixed Pixel Line Grouping (No Adaptive Tolerance) | 0.82 s | 91.40% | -5.70% |
| Native OCR Paragraph Mode | 1.10 s | 89.30% | -7.80% |

*\*Note: The baseline `readtext` method yields a mathematically negligible 0.30% accuracy gain but incurs a 500% latency penalty, rendering it unviable for edge deployment.*

**Analysis:** Relying on native OCR paragraph detection fails to cleanly isolate individual handwritten strokes, causing a 7.80% drop. Similarly, using a fixed-pixel heuristic for line grouping fails on variable-resolution scans, leading to severe under- and over-segmentation (a 5.70% penalty). Our proposed scale-normalized pure detection strategy—coupled with an adaptive tolerance heuristic to handle natural vertical handwriting jitter—preserves bounding box integrity while rescuing the framework from the massive latency bottlenecks associated with full text recognition.

#### 4.3.5. Hierarchical Aggregation
Finally, we evaluate the structural logic of how local patch embeddings are aggregated into a global identity vector for a full multi-line document.

**Table VI: Ablation on Hierarchical Aggregation**

| Configuration Variant | Global Spatial Context | Page-Level Acc. | Δ Accuracy |
| :--- | :--- | :--- | :--- |
| **Proposed: Hierarchical (Patch → Line → Page)** | **Preserved** | **97.10%** | **-** |
| Flat Aggregation (Patch → Page, Skipping Lines) | Spatially Agnostic | 88.00% | -9.10% |

**Analysis:** Bypassing the intermediate line-level pooling phase effectively treats the page as a disorganized "bag of disconnected patches," destroying the spatial geometry of the text. By forcing the network to first define the horizontal stroke dynamics (the line) before averaging those lines into a global identity, the hierarchical pooling strategy preserves spatial context and yields a massive 9.10% improvement in final verification accuracy.
