# 4. Architectural Benchmarking and Ablation Studies

To test our model components efficiently while ensuring fair and rigorous comparisons, all benchmarking and ablation experiments strictly adhered to the hardware configurations, preprocessing pipelines, and training hyperparameters established in Section 3. For the ablation studies, specific modifications were made to the pipeline to empirically justify the selection of each core component.

These experiments were conducted on a subset of 100 writers from the primary dataset, partitioned into 80 writers for training and validation, 20 writers for line-level testing, and a 10-writer subset for page-level tests, in completely disjoint manner.

## 4.1 Architectural Benchmarking

Our primary goal was to build a feature encoder that is highly accurate, lightweight, and fast. To this end, we compared several State-of-the-Art (SOTA) networks using the line-level verification task. Performance was evaluated on a balanced test set of 1,000 line pairs (500 positive and 500 negative samples), with metrics reported at the threshold where Precision equals Recall. We measured model size (Total Parameters) and computational cost (FLOPs for a standard 128 × 128 input patch) using the PyTorch `fvcore` library. We also recorded the average time per training epoch and the real-world inference speed per line, measured precisely using CUDA timers over 100 runs. 

As shown in **Table I**, this comparison highlights two clear solutions. The pretrained FasterNet-T0 emerged as the best SOTA option, achieving the highest accuracy (90.00%) and AUC (0.9676) while remaining highly efficient during inference (3.27 million parameters and 3.44 ms latency). Furthermore, it demonstrated superior training efficiency compared to heavier models like ResNet-18 or ViT-Tiny, converging to its peak performance in just 10 epochs with an average epoch time of 6.85 minutes. 

When designing for strictly limited computational resources, such as edge devices, our custom DPE-Net is the ideal choice. It maintains a strong 83.40% accuracy while drastically shrinking the model size to just 0.177 million parameters and processing each line in only 0.70 ms. This lightweight architecture also translates to exceptionally fast training, requiring only 2.94 minutes per epoch. Because they distinctly address the dual mandates of absolute accuracy and extreme efficiency, we selected both FasterNet-T0 and DPE-Net as the foundational encoders for our entire framework.

**Table I: Architectural Benchmarking on Writer Verification (100-Writer Cohort, Line-Level)**

| Architecture | Paradigm | Total Params | FLOPs (MACs) per patch | Epoch Time | Best / Total Epochs | Inference / Line | Eval Threshold | Line Acc. | Line AUC | Line Prec. | Line Rec. | Line F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pretrained ResNet-18 | Classic CNN | 11.43 M | 568.39 M | 10.16 min | 17 / 20 | 3.45 ms | 0.422 | 89.00% | 96.66% | 89.00% | 89.00% | 0.8900 |
| Pretrained MobileNetV4 (Conv-S) | Mobile CNN | 3.14 M | 60.74 M | 8.68 min | 12 / 20 | 4.13 ms | 0.383 | 86.60% | 93.84% | 86.60% | 86.60% | 0.8660 |
| **Pretrained FasterNet-T0** | **Low-Latency CNN** | **3.27 M** | **109.58 M** | **6.85 min** | **10 / 20** | **3.44 ms** | **0.408** | **90.00%** | **96.76%** | **90.00%** | **90.00%** | **0.9000** |
| Pretrained ViT-Tiny | Pure Transformer | 5.49 M | 349.85 M | 25.62 min | 20 / 20 | 5.82 ms | 0.346 | 89.80% | 96.19% | 89.80% | 89.80% | 0.8980 |
| Pretrained EdgeNeXt-XXS | Hybrid (Edge) | 1.24 M | 64.74 M | 10.85 min | 17 / 20 | 6.06 ms | 0.362 | 89.10% | 95.85% | 89.18% | 89.00% | 0.8909 |
| Pretrained EfficientViT-M0 | Hybrid (Speed) | 2.25 M | 79.13 M | 15.49 min | 17 / 20 | 21.90 ms | 0.483 | 87.80% | 95.22% | 87.80% | 87.80% | 0.8780 |
| Single-Path CNN (MaxPool + Dense) | Custom CNN | 8.61 M | 164.23 M | 14.78 min | 18 / 20 | 1.94 ms | 0.301 | 83.10% | 90.46% | 83.03% | 83.20% | 0.8312 |
| Single-Path CNN (Strided + GAP) | Custom CNN | 0.15 M | 39.49 M | 2.86 min | 18 / 20 | 0.53 ms | 0.240 | 82.60% | 92.08% | 82.60% | 82.60% | 0.8260 |
| Pretrained EdgeNeXt-XXS + Single-Path | Hybrid | 1.42 M | 104.24 M | 19.77 min | 18 / 20 | 10.50 ms | 0.343 | 88.50% | 95.42% | 88.58% | 88.40% | 0.8849 |
| Pretrained FasterNet-T0 + Single-Path | Hybrid | 3.02 M | 148.66 M | 9.31 min | 17 / 20 | 5.82 ms | 0.224 | 89.40% | 96.26% | 89.40% | 89.40% | 0.8940 |
| **Proposed: DPE-Net** | **Custom CNN** | **0.1775 M** | **58.49 M** | **2.94 min** | **18 / 20** | **0.70 ms** | **0.258** | **83.40%** | **92.50%** | **83.40%** | **83.40%** | **0.8340** |






## 4.2 Ablation Studies

To isolate and empirically validate the contribution of each architectural and algorithmic decision, we performed a series of ablation studies. Unless otherwise specified, the "Proposed" baseline framework utilizes: DPE-Net (Mean Pooling), Full Preprocessing (CLAHE + Adaptive Masking), Scale-Normalized Detection with Adaptive Tolerance ($\tau=0.5$), and Hierarchical Aggregation. All line-level experiments utilize 1,000-pair evaluation protocol, while page-level tests utilize 300 balanced pairs (150 positive, 150 negative) derived from the 10-writer subset.

### 4.2.1 Preprocessing and Signal Conditioning

We evaluated the multi-stage signal conditioning pipeline against raw, unconditioned grayscale inputs to determine its impact on feature extraction.

**Table II: Ablation on Line-Level Preprocessing**

| Configuration Variant | Line-Level Accuracy | AUC | $\Delta$ Accuracy |
| :--- | :--- | :--- | :--- |
| **Proposed: Full Pipeline (CLAHE + Adaptive Masking)** | **83.40%** | **0.9250** | **Baseline** |
| Raw Grayscale Inputs (No Preprocessing) | 81.50% | 0.9015 | -1.90% |

As demonstrated in **Table II**, bypassing preprocessing forces the model to process dataset-specific artifacts (e.g., shadows, scanner noise), causing a significant performance drop. The proposed pipeline standardizes stroke geometry and eliminates background noise, yielding a definitive **+1.90%** accuracy increase.

### 4.2.2 Model Architecture and Feature Pooling

Table III details the ablation of the spatial feature extractor, comparing dual-path design, pooling strategies, and spatial flattening methods.

**Table III: Ablation on Model Architecture and Feature Pooling (Line-Level)**

| Configuration Variant | Total Params | Time / Epoch | Inf. / Line | Line-Level Acc. | $\Delta$ Accuracy (vs Proposed) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed: DPE-Net (Mean Pooling)** | **0.1775 M** | **2.94 min** | **0.70 ms** | **83.40%** | **Baseline** |
| Single-Path CNN (Strided Convs + GAP) | 0.1500 M | 2.86 min | 0.53 ms | 82.60% | -0.80% |
| Single-Path CNN (MaxPool + Dense Flattening) | 8.6131 M | 14.78 min | 1.94 ms | 83.10% | -0.30% |
| Proposed DPE-Net utilizing Max-Pooling | 0.1775 M | 2.96 min | 0.83 ms | 81.30% | -2.10% |

Integrating Mean Pooling across the patch sequence proved vastly superior to Max Pooling (+2.10%), confirming that writer identity is best captured mathematically as a continuous average of stylistic traits rather than isolated structural extremes. While the Single-Path CNN (strided + GAP) offered a slightly lighter footprint, adding the parallel dilated branch (DPE-Net) costs only ~27k parameters while boosting accuracy by +0.80%. Furthermore, DPE-Net demonstrated smoother convergence with fewer erratic spikes, reaching a deeper validation minimum of 0.0812 compared to the Single-Path CNN (strided + GAP) (see Appendix A, Figure 12)

### 4.2.3 Page-Level Line Segmentation Ablation

Full-page preprocessing latency is often the primary bottleneck in real-world deployment. We evaluated our Scale-Normalized Detection against alternative OCR-based segmentation strategies, tracking both processing time and downstream biometric accuracy.

**Table IV: Ablation on Page-Level Line Segmentation (Latency vs. Accuracy)**

| Segmentation Strategy | Preprocessing Latency | Total Page Time | Page Acc. | Page AUC | $\Delta$ Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed: Detect + Adaptive Tol ($\tau=0.5$)** | **0.48 s** | **0.51 s** | **96.00%** | **0.9958** | **Baseline** |
| Detect + Adaptive Tol ($\tau=0.1$) | 0.59 s | 0.64 s | 94.67% | 0.9934 | -1.33% |
| Detect + Adaptive Tol ($\tau=0.9$) | 0.46 s | 0.48 s | 94.00% | 0.9891 | -2.00% |
| Baseline OCR (Full CRNN Text Recognition) | 3.05 s | 3.14 s | 93.33% | 0.9913 | -2.67% |
| Native OCR Paragraph Mode (Block Merging) | 2.71 s | 2.72 s | 73.33% | 0.8042 | -22.67% |


As detailed in **Table IV** (where latency is averaged per page on a single NVIDIA RTX 3050, and total time encompasses both segmentation and downstream DPE-Net inference), the results highlight three critical deployment insights. First, our pure-detection method avoids the severe latency penalty of OCR, operating over 6x faster than full CRNN semantic decoding (0.51 s vs. 3.14 s) while actually yielding a +2.67% increase in downstream accuracy. This empirically proves that semantic decoding is an unnecessary computational burden for purely biometric tasks. Second, relying on native OCR paragraph grouping resulted in a catastrophic 22.67% accuracy drop. This native mode aggressively merges distinct lines into massive vertical blocks, destroying the horizontal spatial consistency required by the patch encoder. Finally, optimizing the vertical grouping tolerance ($\tau$) was crucial to segmentation performance; a strict $\tau = 0.1$ caused over-segmentation, while a loose $\tau = 0.9$ caused under-segmentation. The proposed $\tau = 0.5$ successfully preserved perfect single-line continuity, ultimately maximizing page-level accuracy.








### 4.2.4 Local-to-Global Feature Integration Ablation

Traditional page-level document analysis typically relies on "flat" architectures that directly integrate local features into a global representation (Patch $\rightarrow$ Page). To evaluate the necessity of explicit line segmentation while generating page-level encodings, we benchmarked our proposed hierarchical integration (Patch $\rightarrow$ Line $\rightarrow$ Page) against three such flat baselines using 300 balanced (150 positive, 150 negative) page-level pairs. The quantitative results of this comparison are detailed in **Table V**.


**Table V: Ablation on Hierarchical Aggregation (Page-Level)**

| Aggregation Strategy | Precision | Recall | F1-Score | Accuracy | $\Delta$ Accuracy | PR=RC Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed: Hierarchical (Patch $\rightarrow$ Line $\rightarrow$ Page)** | **96.00%** | **96.00%** | **96.00%** | **96.00%** | **-** | **0.138500** |
| Flat (Patch $\rightarrow$ Page): CNN + Self-Attention + GeM | 92.67% | 92.67% | 92.67% | 92.67% | -3.33% | 0.157500 |
| Flat (Patch $\rightarrow$ Page): CNN + Transformer ([CLS] Token) | 85.33% | 85.33% | 85.33% | 85.33% | -10.67% | 0.114000 |
| Flat (Patch $\rightarrow$ Page): CNN + NetVLAD | 53.99% | 94.67% | 68.76% | 57.00% | -39.00% | 0.000500* |

*\*Note: For the NetVLAD architecture, Precision and Recall did not converge at any discrete point; the closest mathematical approximation is reported.*


The flat baselines bypass segmentation entirely, treating the document as an unordered "bag of patches" by sampling 32 ink-rich patches via a sliding window. Because these randomly sampled patches lack a spatial sequence—whereas continuous lines inherently preserve spatial knowledge and better capture the writer's stylistic flow—they struggle to separate structural noise from true handwriting. CNN + NetVLAD fundamentally failed (57.00% accuracy) at clustering these patches, while a Transformer utilizing a `[CLS]` token gathered context more effectively (85.33%). The most robust flat baseline, CNN + Self-Attention + GeM Pooling (92.67%), successfully emphasized salient strokes and processed full pages rapidly.

However, the proposed hierarchical architecture proved strictly superior across all metrics, peaking at 96.00% accuracy. This performance gap highlights the fundamental flaw of flat sampling: a global "bag of patches" inevitably captures macro-level layout noise and inter-line spacing artifacts. By segmenting the document into distinct lines first, the hierarchical approach introduces a critical structural prior. It forces the network to evaluate contiguous text, systematically filtering out formatting variations to extract a pure biometric representation of the writer's penmanship.





### 4.2.5 Unsupervised Page Clustering Ablation

For the document clustering pipeline, the foundational DPE-Net (Mean Pooling) was fully trained on the primary 371-writer training and validation cohort. We evaluated clustering algorithms using disjoint writer sets: writers 372–403 were used for hyperparameter tuning, and writers 404–415 were isolated for final evaluation.

**Table VI: Ablation on Unsupervised Page Clustering Algorithms**

| Algorithm | Hyperparameters | Tuning ACC | Tuning ARI | Eval Clusters | Eval ARI | Eval NMI | Eval F1-Score | Eval Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed: Agglomerative** | Distance Thresh: 0.110 | **99.21%** | **0.9532** | 23 | **0.7630** | **0.8658** | **0.7875** | **95.60%** |
| DBSCAN | eps: 0.060, min_samples: 2 | 98.36% | 0.9069 | 9 (+17 Noise) | 0.5847 | 0.7791 | 0.6415 | 90.86% |

While both algorithms performed exceptionally well on the tuning set, Agglomerative Clustering demonstrated significantly better scaling to the larger, unseen evaluation set, maintaining a high Adjusted Rand Index (0.7630) and overall Accuracy (95.60%). Conversely, DBSCAN's performance degraded sharply because its density-based approach classified 17 out of 138 evaluation pages as unclusterable "noise." Discarding over 12% of the dataset is unacceptable in real-world document sorting. This ablation proves that for high-dimensional, $L_2$-normalized biometric embeddings, distance-based hierarchical merging (Agglomerative) is inherently more robust and comprehensive for grouping writer styles than density-reachability metrics.

### 4.2.6 Sequential Multi-Writer Segmentation Ablation

To evaluate the sequential multi-writer segmentation pipeline, we again utilized the fully trained DPE-Net. Hyperparameter tuning was conducted on 196 synthesized multi-writer pages (derived from writers 395–420), with final evaluation performed on 97 synthesized multi-writer pages (from writers 421–435).

**Table VII: Ablation on Sequential Multi-Writer Segmentation Pipelines**

| Segmentation Algorithm | Optimal Distance / Eps | Optimal Window | Tuning SER (196 Pages) | Eval SER (97 Pages) | Absolute Sequence Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed: DBSCAN** | **0.090** | **5** | **0.0931** | **0.0868** | **75.26%** |
| Agglomerative | 0.220 | 7 | 0.0906 | 0.1916 | 72.16% |

The most striking result is the superior generalization of the proposed DBSCAN model (optimized at `eps = 0.090` and `min_samples = 3`). While Agglomerative Clustering slightly edged out DBSCAN on the tuning set (0.0906 vs. 0.0931 Sequence Error Rate), its performance degraded significantly on the unseen evaluation set (0.1916 SER). In contrast, DBSCAN maintained high stability, actually improving to a remarkable 0.0868 SER on the evaluation pages. Furthermore, DBSCAN proved to be the superior framework for absolute chronological reconstruction, perfectly mapping the correct writer transitions on 75.26% of the test pages. Finally, DBSCAN optimized at a tighter temporal smoothing window (5 compared to the baseline's 7), indicating that its density-reachability metric inherently produces more contiguous, stable line-level assignments without relying on aggressive post-hoc smoothing to correct hallucinated transitions.


