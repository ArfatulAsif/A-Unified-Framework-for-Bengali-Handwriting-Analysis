# 3. Methodology

In this section, we detail the complete pipeline of our proposed writer identification framework. We begin by outlining our dataset preparation and preprocessing strategies, followed by the core architecture of our Hierarchical Patch Encoder, the comparative State-of-the-Art (SOTA) architectures considered, and our training objectives and hyperparameter setups. Finally, we break down the framework into four distinct operational pipelines: Line and Page-Level Inference (Verification), Writer-Based Document Retrieval, Unsupervised Document Clustering, and Sequential Multi-Writer Segmentation.

All computational tasks, including data preprocessing and model training, were executed using Python 3.9.13 on a single NVIDIA RTX 3050 GPU, utilizing PyTorch 2.8.0, CUDA 12.8, and cuDNN 8.9.7.

## 3.1 Dataset Collection and Preparation

To construct a comprehensive, highly variable, and inclusive dataset, we aggregated data from three primary sources:

1.  **BN-HTRd:** Provided existing line-segmented images and full handwritten pages from 237 distinct writers.
2.  **WBSUBNdb\_text:** Contributed handwritten pages from 188 writers. We manually processed these pages through our line-segmentation pipeline to create dedicated line-level datasets, visually evaluating and verifying each segmented line to ensure high ground-truth quality.
3.  **Custom Robustness Dataset:** We compiled an additional, highly challenging set featuring 10 specialized writers. This custom subset was specifically curated to introduce severe real-world impurities, including low-quality mobile phone scans, heavy shadows, and varied camera orientations.

Throughout the aggregation process, we thoroughly reviewed the data to ensure it accurately reflected the unpredictable nature of real-world physical documents, explicitly keeping variations such as exceptionally long and short lines, diverse page colors, and curved or slanted handwriting.

**Dataset Summary:**

  * **Total Writers:** 435
  * **Total Handwritten Pages:** 2,825 (Average of \~7 pages per writer)
  * **Total Segmented Lines:** 29,268 (Average of \~70 lines per writer)

## 3.2 Writer-Disjoint Split and Evaluation Protocol

To ensure our model learns generalized, writer-independent style features rather than memorizing the specific handwriting traits of the training set, we adhered to a strict, zero-shot, writer-disjoint evaluation protocol (illustrated in **Figure 1**). Maintaining this open-set approach throughout the entire pipeline is critical for proving that our framework functions reliably on entirely unseen writers in real-world forensic scenarios.

The overarching dataset of 435 writers was strategically partitioned into the following non-overlapping subsets:

  * **Patch Encoder Training (Writers 1–371):** The segmented handwriting lines from these 371 writers were used exclusively to train the core CNN patch encoder.
  * **Verification Tuning (Writers 372–403):** Line and page-level handwriting from these writers were used for comparative evaluation. This subset served to determine the optimal cosine distance threshold for authenticating whether two handwriting samples belong to the same writer.
  * **Retrieval Evaluation (Writers 404–415):** Page-level handwriting from these unseen writers was used to evaluate the Writer-Based Document Retrieval pipeline, utilizing the optimal distance threshold established during the verification tuning phase.
  * **Clustering Tuning (Writers 372–403):** Page-level data from this subset was reused to tune the unsupervised document clustering algorithms and establish the optimal hyperparameters (such as `eps`, `min_samples`, and `distance_threshold`).
  * **Clustering Evaluation (Writers 404–415):** Page-level handwriting from these unseen writers was used to strictly evaluate the final document clustering pipeline using the locked hyperparameters.
  * **Multi-Writer Segmentation (Writers 395–435):** To develop and test the sequential writer diarization pipeline, we synthesized multi-writer documents by cropping and vertically merging segments from different writers, yielding specialized 1-writer, 2-writer, 3-writer, and 4-writer pages.
      * **Segmentation Tuning (Writers 395–420):** Used to tune the clustering and post-processing smoothing windows.
      * **Segmentation Evaluation (Writers 421–435):** Used for the final, strict evaluation of the pipeline's Sequence Error Rate (SER).

**Standardized Ablation Cohort:**
Finally, to conduct fair State-of-the-Art (SOTA) comparisons and detailed ablation studies without exhausting our primary test sets, we isolated a standardized mini-cohort of 100 writers. This cohort was strictly partitioned into 80 train/validation writers and 20 test writers for line-level tasks, with 10 of those test writers reserved for page-level evaluations.



<img src="images/dataset.png">
