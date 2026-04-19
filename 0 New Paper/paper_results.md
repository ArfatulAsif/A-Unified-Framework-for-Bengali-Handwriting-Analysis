# 5. Final System Results

To evaluate the proposed framework, both FasterNet-T0 and DPE-Net were trained from scratch on the primary line-level dataset comprising 371 writers, partitioned into 297 for training and 74 for validation (an 80/20 writer disjoint split). The training dynamics, including convergence points and computational costs, are summarized in **Table VIII**. The complete training loss trajectory for DPE-Net is provided in **Appendix Figure 11**.

**Table VIII: Training Dynamics and Convergence Metrics**

| Architecture | Best Epoch / Total | Best Validation Loss | Avg. Epoch Time |
| :--- | :--- | :--- | :--- |
| **DPE-Net** | 36 / 46 | 0.0416 | 17.06 min |
| **FasterNet-T0** | 36 / 46 | 0.0227 | 45.35 min |

## 5.1 Line and Page-Level Evaluation Results

The final evaluation was conducted on a completely disjoint cohort of 32 unseen writers (IDs 372–403). Following the protocol established in Section 3.8, all metrics were computed at the Break-Even Point (where Precision equals Recall). The dynamic threshold sweep diagrams for these evaluations are provided in **Appendix Figure 12**.

To rigorously test both local stroke extraction and global document aggregation, the evaluation was performed sequentially at both the line level (utilizing a strictly balanced set of 1,000 line pairs) and the page level (utilizing a balanced set of 400 full-document pairs). The systemic performance metrics for both stages, including the end-to-end processing latency for full pages, are summarized in **Table IX**.

**Table IX: Final Line and Page-Level Writer Verification Results (32 Unseen Writers)**

| Architecture | Evaluation Scope | Accuracy | AUC | Precision | Recall | F1-Score | Eval Threshold | Avg Total Time / Page |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DPE-Net** | Line-Level (1000 Pairs) | 90.10% | 0.9669 | 0.9018 | 0.9000 | 0.9009 | 0.3148 | - |
| **DPE-Net** | Page-Level (400 Pairs) | 95.00% | 0.9921 | 0.9500 | 0.9500 | 0.9500 | 0.1905 | 0.5163 s |
| **FasterNet-T0** | Line-Level (1000 Pairs) | 93.80% | 0.9861 | 0.9380 | 0.9380 | 0.9380 | 0.3680 | - |
| **FasterNet-T0** | Page-Level (400 Pairs) | 96.00% | 0.9949 | 0.9600 | 0.9600 | 0.9600 | 0.2525 | 0.5969 s |

## 5.2 Writer-Based Document Retrieval Evaluation Results

The writer-based document retrieval evaluation (1:N search) was conducted on a cohort of 138 full document pages authored by 12 distinct writers (404–415 writers). Following the Leave-One-Out (LOO) ranking protocol established in Section 3.9, each page was iteratively isolated and utilized as a query against the remaining search gallery. 

The retrieval performance was assessed by calculating the Cosine Distance between the global page embeddings to sort the ranked lists. The final retrieval metrics are summarized in **Table X**.

**Table X: Writer-Based Document Retrieval Results (138 Pages, 12 Writers)**

| Architecture | Top-1 Accuracy | Top-5 Accuracy | Mean Average Precision (mAP) |
| :--- | :--- | :--- | :--- |
| **DPE-Net** | 94.20% | 98.55% | 91.19% |
| **FasterNet-T0** | 100.00% | 100.00% | 96.60% |

## 5.3 Handwriting-Based Document Clustering Evaluation Results

To evaluate the unsupervised clustering capabilities of the framework, Agglomerative Hierarchical Clustering (with average linkage) was applied to the global page-level embeddings. As established in Section 3.10, the optimal distance threshold ($\tau_{cluster}$) for each architecture was determined via a parameter sweep on a disjoint tuning cohort of 239 pages from 32 writers (372–403 writers). This sweep identified the optimal stopping criteria as $\tau_{cluster} = 0.150$ for DPE-Net and $\tau_{cluster} = 0.110$ for FasterNet-T0.

The final clustering evaluation was executed on the isolated test cohort comprising 138 pages across 12 ground-truth writers (404–415). The results, encompassing both global partitioning metrics (ARI, NMI) and pairwise assignment metrics, are presented in **Table XI**.

**Table XI: Unsupervised Document Clustering Results (138 Pages, 12 Ground-Truth Writers)**

| Architecture | Distance Threshold ($\tau_{cluster}$) | Clusters Formed | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) | Pairwise Accuracy | Pairwise Precision | Pairwise Recall | Pairwise F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DPE-Net** | 0.150 | 19 | 0.7865 | 0.9000 | 96.03% | 0.7857 | 0.8328 | 0.8086 |
| **FasterNet-T0** | 0.110 | 22 | 0.9198 | 0.9445 | 98.64% | 1.0000 | 0.8644 | 0.9272 |

FasterNet-T0 partitioned the evaluation corpus into 22 distinct clusters, resulting in a 0.9198 ARI and a 98.64% pairwise accuracy. DPE-Net partitioned the dataset into 19 clusters, resulting in a 0.7865 ARI and a 96.03% pairwise accuracy.

## 5.4 Sequential Multi-Writer Segmentation Evaluation Results

To evaluate the sequential multi-writer segmentation pipeline, DBSCAN clustering was applied to the chronological line embeddings following the protocol established in Section 3.11. 

Hyperparameter tuning was conducted on a tuning cohort of 196 synthesized multi-writer pages to determine the optimal Maximum Neighborhood Distance ($\epsilon$), Minimum Samples ($\mu_{samples}$), and Smoothing Window Size ($w_{size}$) for each spatial encoder. The final evaluation was subsequently executed on a completely disjoint test set of 97 synthesized multi-writer pages. The sequence metrics, including the Sequence Error Rate (SER) and Absolute Sequence Accuracy, are presented in **Table XII**.

**Table XII: Sequential Multi-Writer Segmentation Results (196 Tuning Pages, 97 Evaluation Pages)**

| Architecture | Optimal $\epsilon$ | Minimum Samples | Window Size | Tuning SER | Eval SER | Absolute Sequence Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DPE-Net** | 0.09 | 3 | 5 | 0.0931 | 0.0868 | 75.26% |
| **FasterNet-T0** | 0.10 | 2 | 5 | 0.0825 | 0.1168 | 79.38% |

FasterNet-T0 achieved an Absolute Sequence Accuracy of 79.38% and a final evaluation SER of 0.1168. DPE-Net recorded an evaluation SER of 0.0868 and an Absolute Sequence Accuracy of 75.26%.

***




